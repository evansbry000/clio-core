/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * GPU benchmark kernels for CTE PutBlob/GetBlob.
 *
 * Each GPU block acts as an independent worker thread, issuing
 * AsyncPutBlob / AsyncGetBlob via the Client API (NewTask + Send ->
 * SendGpu -> gpu2gpu queue). The GPU orchestrator processes tasks
 * on separate blocks.
 *
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <thread>
#include <chrono>

/**
 * Benchmark parameters passed to the kernel.
 */
struct GpuBenchParams {
  chi::PoolId pool_id;
  wrp_cte::core::TagId tag_id;
  chi::u64 io_size;
  int io_count;      // Per-block I/O count
  int mode;          // 0=Put, 1=Get, 2=PutGet
};

/**
 * Per-block result written to pinned memory.
 */
struct GpuBenchResult {
  int status;              // 1=success, negative=error
  long long elapsed_ns;    // Nanoseconds elapsed for this block
  long long send_clocks;   // Total clocks in AsyncPutBlob/AsyncGetBlob (send side)
  long long wait_clocks;   // Total clocks in future.Wait() (receive side)
};

/**
 * GPU kernel: each block runs io_count Put and/or Get operations.
 * Only thread 0 of each block does work (matches orchestrator pattern).
 */
__global__ void gpu_bench_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    GpuBenchParams params,
    GpuBenchResult *results,
    chi::u32 num_blocks) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  if (threadIdx.x != 0) return;

  int block_id = blockIdx.x;
  GpuBenchResult *my_result = &results[block_id];

  wrp_cte::core::Client client(params.pool_id);

  // Allocate put/get buffers from GPU device memory
  auto put_full = CHI_IPC->AllocateDeviceData(params.io_size);
  if (put_full.IsNull()) { my_result->status = -10; return; }
  auto get_full = CHI_IPC->AllocateDeviceData(params.io_size);
  if (get_full.IsNull()) { my_result->status = -11; return; }

  // Fill put buffer with pattern
  char *put_buf = put_full.ptr_;
  char *get_buf = get_full.ptr_;
  memset(put_buf, 0xAB, params.io_size);
  memset(get_buf, 0x00, params.io_size);

  // Build ShmPtr from the FullPtr (carries allocator ID + offset)
  hipc::ShmPtr<> put_shm(put_full.shm_);
  hipc::ShmPtr<> get_shm(get_full.shm_);

  // Build per-block blob name: "blob_0", "blob_1", etc.
  char blob_name[32];
  int name_len = 0;
  const char *prefix = "blob_";
  for (int i = 0; prefix[i]; ++i) blob_name[name_len++] = prefix[i];
  // Convert block_id to string digits
  char digits[16];
  int num_digits = 0;
  int tmp = block_id;
  if (tmp == 0) { digits[num_digits++] = '0'; }
  else { while (tmp > 0) { digits[num_digits++] = '0' + (tmp % 10); tmp /= 10; } }
  for (int i = num_digits - 1; i >= 0; --i) blob_name[name_len++] = digits[i];
  blob_name[name_len] = '\0';

  // Use GPU clock for timing
  long long start_clock = clock64();
  long long total_send = 0, total_wait = 0;

  if (params.mode == 0 || params.mode == 2) {
    // Put phase
    for (int i = 0; i < params.io_count; ++i) {
      long long t0 = clock64();
      auto future = client.AsyncPutBlob(
          params.tag_id, blob_name,
          chi::u64(0), params.io_size,
          put_shm, 0.5f,
          wrp_cte::core::Context(),
          chi::u32(0),
          chi::PoolQuery::Local());
      long long t1 = clock64();
      total_send += (t1 - t0);
      if (future.IsNull()) {
        my_result->status = -2;
        return;
      }
      future.Wait();
      long long t2 = clock64();
      total_wait += (t2 - t1);
    }
  }

  if (params.mode == 1 || params.mode == 2) {
    // Get phase
    for (int i = 0; i < params.io_count; ++i) {
      long long t0 = clock64();
      auto future = client.AsyncGetBlob(
          params.tag_id, blob_name,
          chi::u64(0), params.io_size,
          chi::u32(0),
          get_shm,
          chi::PoolQuery::Local());
      long long t1 = clock64();
      total_send += (t1 - t0);
      if (future.IsNull()) {
        my_result->status = -3;
        return;
      }
      future.Wait();
      long long t2 = clock64();
      total_wait += (t2 - t1);
    }
  }

  long long end_clock = clock64();
  my_result->elapsed_ns = end_clock - start_clock;
  my_result->send_clocks = total_send;
  my_result->wait_clocks = total_wait;
  my_result->status = 1;
}

/**
 * Host entry point: sets up GPU backends, launches benchmark kernel,
 * and collects timing results.
 *
 * @param pool_id      CTE core pool ID
 * @param tag_id       Tag ID for blob operations
 * @param io_size      Size of each I/O in bytes
 * @param io_count     Number of I/Os per block
 * @param mode         0=Put, 1=Get, 2=PutGet
 * @param num_blocks   Number of GPU client blocks
 * @param num_threads  Threads per block (only thread 0 does work)
 * @param results      Output array (num_blocks entries, pinned memory)
 * @return 0 on success, negative on error
 */
extern "C" int run_gpu_bench(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u64 io_size,
    int io_count,
    int mode,
    int num_blocks,
    int num_threads,
    GpuBenchResult *results) {

  // Create GPU memory backend for client kernel allocations
  // Backend is split into (num_blocks * num_threads) BuddyAllocator partitions.
  // Only thread 0 per block allocates, but all threads need a partition.
  hipc::MemoryBackendId backend_id(20, 0);
  hipc::GpuShmMmap gpu_backend;
  size_t per_thread = std::max(
      static_cast<size_t>(4) * 1024 * 1024,
      static_cast<size_t>(8) * io_size);
  size_t backend_size = std::max(
      static_cast<size_t>(64) * 1024 * 1024,
      static_cast<size_t>(num_blocks * num_threads) * per_thread);
  if (!gpu_backend.shm_init(backend_id, backend_size,
                             "/cte_gpu_bench", 0))
    return -100;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // GPU heap for serialization: 1 MB per block (min 16 MB)
  hipc::MemoryBackendId heap_id(21, 0);
  hipc::GpuMalloc gpu_heap;
  size_t heap_size = std::max(
      static_cast<size_t>(16) * 1024 * 1024,
      static_cast<size_t>(num_blocks) * 1024 * 1024);
  if (!gpu_heap.shm_init(heap_id, heap_size, "", 0))
    return -102;

  chi::IpcManagerGpuInfo gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_heap_backend = gpu_heap;

  // Initialize results
  for (int i = 0; i < num_blocks; ++i) {
    results[i].status = 0;
    results[i].elapsed_ns = 0;
  }

  GpuBenchParams params;
  params.pool_id = pool_id;
  params.tag_id = tag_id;
  params.io_size = io_size;
  params.io_count = io_count;
  params.mode = mode;

  // Set GPU stack size for deep serialization paths (PutGet needs more)
  cudaDeviceSetLimit(cudaLimitStackSize, 32768);

  // Pause orchestrator, launch client kernel, resume
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_bench_kernel<<<num_blocks, num_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, params, results,
      static_cast<chi::u32>(num_blocks));

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  CHI_IPC->ResumeGpuOrchestrator();

  // Poll pinned memory for all blocks to complete
  int timeout_us = 30000000;  // 30 seconds
  int elapsed_us = 0;
  while (elapsed_us < timeout_us) {
    bool all_done = true;
    for (int i = 0; i < num_blocks; ++i) {
      if (results[i].status == 0) {
        all_done = false;
        break;
      }
    }
    if (all_done) break;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }

  hshm::GpuApi::DestroyStream(stream);

  // Check for errors
  for (int i = 0; i < num_blocks; ++i) {
    if (results[i].status != 1) {
      return results[i].status == 0 ? -4 : results[i].status;
    }
  }

  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
