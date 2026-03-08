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
 * GPU Runtime Benchmark — GPU kernel and CUDA wrapper
 *
 * Contains only the GPU kernel and a thin kernel-launch wrapper.
 * All IpcManager access is in bench_gpu_runtime.cc (g++-compiled) to avoid
 * ODR layout mismatches between nvcc and g++ views of IpcManager.
 *
 * Only thread 0 of each block submits tasks (matches the single-thread
 * pattern proven in unit tests). Parallelism comes from client_blocks:
 * each block independently submits total_tasks sequential tasks. Block 0
 * thread 0 writes d_done after its tasks complete to signal the CPU.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/ipc_manager.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/MOD_NAME/autogen/MOD_NAME_methods.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>

#include <chrono>
#include <thread>

namespace chi_bench {

/**
 * GPU client benchmark kernel.
 *
 * Each block's thread 0 initializes its IpcManager (via
 * CHIMAERA_GPU_ORCHESTRATOR_INIT for per-block backend partitioning), then
 * submits total_tasks tasks sequentially via AsyncGpuSubmit + Wait(). Other
 * threads in the block are idle. Block 0 thread 0 writes d_done after its
 * loop completes.
 *
 * @param gpu_info   IpcManagerGpuInfo with backend and GPU→GPU queue
 * @param pool_id    Pool ID of the MOD_NAME container
 * @param num_blocks Number of blocks in the grid (for per-block backend slice)
 * @param total_tasks Total tasks for thread 0 of each block to submit
 * @param d_done     Pinned host flag set to 1 by block 0, thread 0 on finish
 */
__global__ void gpu_bench_client_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done) {
  // Partition backend per block; initialize block-local IpcManager
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  // Only thread 0 of each block does work (proven pattern from unit tests)
  if (threadIdx.x != 0) {
    return;
  }

  chimaera::MOD_NAME::Client client(pool_id);

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), 0, i);
    future.Wait();
  }

  // Block 0 thread 0 signals completion to the CPU
  if (blockIdx.x == 0) {
    __threadfence_system();
    *d_done = 1;
  }
}

}  // namespace chi_bench

/**
 * Poll the pinned done flag until set or timeout.
 */
static bool PollDone(volatile int *d_done, int timeout_us) {
  int elapsed_us = 0;
  while (*d_done == 0 && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }
  return *d_done == 1;
}

/**
 * Run the GPU runtime latency benchmark.
 *
 * All IpcManager access (RegisterGpuAllocator, GetGpuToGpuQueue,
 * SetGpuOrchestratorBlocks) is done via non-inline methods defined in
 * ipc_manager.cc (g++-compiled) to avoid ODR layout mismatches.
 *
 * @param pool_id        Pool ID of the MOD_NAME container
 * @param method_id      (unused; kept for ABI compatibility)
 * @param rt_blocks      GPU work orchestrator block count
 * @param rt_threads     GPU work orchestrator threads per block
 * @param client_blocks  GPU client kernel block count
 * @param client_threads (unused; only thread 0 per block does work)
 * @param batch_size     (unused; kept for ABI compatibility)
 * @param total_tasks    Total sequential tasks per block's thread 0
 * @param out_elapsed_ms Output: elapsed wall-clock time in ms
 * @return 0 on success, negative error code on failure
 */
extern "C" int run_gpu_bench_latency(
    chi::PoolId pool_id,
    chi::u32 method_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 batch_size,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Use non-inline SetGpuOrchestratorBlocks to avoid ODR layout mismatch
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Allocate primary GPU backend (GpuShmMmap, pinned host): 10 MB per block.
  // Used by ArenaAllocator (HSHM_DEFAULT_ALLOC_GPU_T) for FutureShm allocation.
  // Partitioned per block by CHIMAERA_GPU_ORCHESTRATOR_INIT.
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(100, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "/bench_gpu_runtime", 0)) {
    return -1;
  }

  // Use non-inline RegisterGpuAllocator to avoid ODR layout mismatch
  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Allocate GPU heap backend (GpuMalloc, device memory): 4 MB per block.
  // Used by BuddyAllocator (CHI_GPU_HEAP_T) for serialization scratch buffers.
  // BuddyAllocator supports individual free, avoiding arena exhaustion.
  constexpr size_t kPerBlockHeapBytes = 4 * 1024 * 1024;
  size_t heap_backend_size = static_cast<size_t>(client_blocks) * kPerBlockHeapBytes;

  hipc::MemoryBackendId heap_backend_id(101, 0);
  hipc::GpuMalloc gpu_heap_backend;
  if (!gpu_heap_backend.shm_init(heap_backend_id, heap_backend_size,
                                  "/bench_gpu_runtime_heap", 0)) {
    return -1;
  }

  // Build IpcManagerGpuInfo: GPU→GPU path only (no CPU queues needed)
  // Use non-inline GetGpuToGpuQueue to avoid ODR layout mismatch
  chi::IpcManagerGpu gpu_info;
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_heap_backend = gpu_heap_backend;
  gpu_info.gpu2cpu_queue = nullptr;
  gpu_info.cpu2gpu_queue = nullptr;
  gpu_info.gpu2gpu_queue = CHI_IPC->GetGpuToGpuQueue(0);

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();

  // Pause orchestrator to free SMs, launch client kernel, then resume
  CHI_IPC->PauseGpuOrchestrator();
  cudaGetLastError();  // Clear any sticky CUDA errors

  chi_bench::gpu_bench_client_kernel<<<
      client_blocks, 1, 0,   // 1 thread per block (proven pattern)
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  // Resume orchestrator so it can process the GPU→GPU tasks
  CHI_IPC->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  // Poll for block 0 completion (60 s timeout)
  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  // Pause orchestrator before cleanup
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;  // -4 = timeout
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
