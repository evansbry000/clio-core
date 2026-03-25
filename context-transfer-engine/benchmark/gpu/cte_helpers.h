/**
 * cte_helpers.h — Shared CTE boilerplate for workload benchmarks
 */
#ifndef BENCH_GPU_CTE_HELPERS_H
#define BENCH_GPU_CTE_HELPERS_H

#include <hermes_shm/constants/macros.h>

// Declare the alloc kernel from wrp_cte_gpu_bench.cc (compiled in both passes)
extern __global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr);

#if HSHM_IS_HOST

#include "workload.h"
#include <thread>

struct CteGpuContext {
  hipc::MemoryBackendId data_id{200, 0};
  hipc::MemoryBackendId scratch_id{201, 0};
  hipc::MemoryBackendId heap_id{202, 0};
  hipc::GpuMalloc data_backend;
  hipc::GpuMalloc scratch_backend;
  hipc::GpuMalloc heap_backend;
  hipc::FullPtr<char> array_ptr;
  hipc::AllocatorId data_alloc_id;
  chi::IpcManagerGpu gpu_info;
  int *d_done = nullptr;
  volatile int *d_progress = nullptr;
  uint32_t total_warps = 0;
  bool valid = false;

  int init(uint64_t data_bytes, uint32_t num_warps) {
    total_warps = num_warps;
    CHI_IPC->PauseGpuOrchestrator();

    size_t data_size = data_bytes + 4 * 1024 * 1024;
    if (!data_backend.shm_init(data_id, data_size, "", 0)) {
      CHI_IPC->ResumeGpuOrchestrator(); return -1;
    }
    size_t scratch_size = (size_t)num_warps * 1024 * 1024;
    if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
      CHI_IPC->ResumeGpuOrchestrator(); return -1;
    }
    size_t heap_size = (size_t)num_warps * 1024 * 1024;
    if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
      CHI_IPC->ResumeGpuOrchestrator(); return -1;
    }

    hipc::FullPtr<char> *d_ptr;
    cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gpu_putblob_alloc_kernel<<<1, 1>>>(
        static_cast<hipc::MemoryBackend &>(data_backend),
        data_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator(); return -2;
    }
    array_ptr = *d_ptr;
    cudaFreeHost(d_ptr);

    data_alloc_id = hipc::AllocatorId(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_,
                                   data_backend.data_capacity_);

    gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;

    cudaMallocHost(&d_done, sizeof(int));
    *d_done = 0;
    cudaMallocHost((void**)&d_progress, sizeof(int) * num_warps);
    memset((void*)d_progress, 0, sizeof(int) * num_warps);

    if (scratch_backend.data_)
      cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    if (heap_backend.data_)
      cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    valid = true;
    return 0;
  }

  bool resume_and_poll(int timeout_sec) {
    CHI_IPC->ResumeGpuOrchestrator();
    auto *orch = static_cast<chi::gpu::WorkOrchestrator *>(
        CHI_IPC->gpu_orchestrator_);
    auto *ctrl = orch ? orch->control_ : nullptr;
    if (ctrl) {
      int wait_ms = 0;
      while (ctrl->running_flag == 0 && wait_ms < 5000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++wait_ms;
      }
    }
    int64_t timeout_us = (int64_t)timeout_sec * 1000000;
    int64_t elapsed = 0;
    while (__atomic_load_n(d_done, __ATOMIC_ACQUIRE) < (int)total_warps &&
           elapsed < timeout_us) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      elapsed += 100;
    }
    return __atomic_load_n(d_done, __ATOMIC_ACQUIRE) >= (int)total_warps;
  }

  void cleanup() {
    if (d_done) { cudaFreeHost(d_done); d_done = nullptr; }
    if (d_progress) { cudaFreeHost((void*)d_progress); d_progress = nullptr; }
    valid = false;
  }
};

#endif  // HSHM_IS_HOST
#endif  // BENCH_GPU_CTE_HELPERS_H
