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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_

#include "chimaera/types.h"
#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"

namespace chi {
namespace gpu {

/**
 * Lightweight RunContext for GPU-side task execution.
 * Unlike the host RunContext, this has no STL members, no coroutines,
 * and is fully GPU-compatible.
 */
struct GpuRunContext {
  u32 block_id_;
  u32 thread_id_;

  HSHM_GPU_FUN GpuRunContext() : block_id_(0), thread_id_(0) {}
  HSHM_GPU_FUN GpuRunContext(u32 block_id, u32 thread_id)
      : block_id_(block_id), thread_id_(thread_id) {}
};

/**
 * GPU-side container base class
 *
 * Unlike CPU containers, GPU containers:
 * - Use no STL members (must be device-compatible)
 * - Have no coroutine support
 * - Use HSHM_GPU_FUN for all methods
 * - Use DynamicSchedule instead of ScheduleTask for routing
 */
class Container {
 public:
  PoolId pool_id_;
  u32 container_id_;
  HSHM_DEFAULT_ALLOC_GPU_T *gpu_alloc_ = nullptr;  /**< Set by worker before dispatch */

  HSHM_GPU_FUN Container() : container_id_(0), gpu_alloc_(nullptr) {}
  HSHM_GPU_FUN virtual ~Container() = default;

  /**
   * Initialize the GPU container
   * @param pool_id Pool identifier
   * @param container_id Container ID (typically node_id)
   */
  HSHM_GPU_FUN virtual void Init(const PoolId &pool_id, u32 container_id) {
    pool_id_ = pool_id;
    container_id_ = container_id;
  }

  /**
   * Execute a task method on the GPU
   * @param method Method ID to execute
   * @param task_ptr Full pointer to the task
   * @param rctx GPU run context
   */
  HSHM_GPU_FUN virtual void Run(u32 method, hipc::FullPtr<Task> task_ptr,
                                 GpuRunContext &rctx) = 0;

  /**
   * Dynamic scheduling for GPU tasks
   * Returns a PoolQuery indicating where the task should be routed.
   * Default: use the task's existing pool_query_ (local execution)
   * @param method Method ID
   * @param task_ptr Full pointer to the task
   * @return PoolQuery for routing decision
   */
  HSHM_GPU_FUN virtual PoolQuery DynamicSchedule(
      u32 method, hipc::FullPtr<Task> task_ptr) {
    return task_ptr->pool_query_;
  }

  /**
   * Allocate and deserialize a task from a local archive.
   * Called by gpu::Worker after ShmTransport::Recv populates the archive.
   *
   * @param method Method ID identifying the task type
   * @param archive LocalLoadTaskArchive containing serialized input
   * @return FullPtr to the deserialized task, or null on failure
   */
  HSHM_GPU_FUN virtual hipc::FullPtr<Task> LocalAllocLoadTask(
      u32 method, LocalLoadTaskArchive &archive) = 0;

  /**
   * Serialize task output into a local archive.
   * Called by gpu::Worker before ShmTransport::Send writes to the ring buffer.
   *
   * @param method Method ID identifying the task type
   * @param archive LocalSaveTaskArchive to write output into
   * @param task FullPtr to the completed task
   */
  HSHM_GPU_FUN virtual void LocalSaveTask(
      u32 method, LocalSaveTaskArchive &archive,
      const hipc::FullPtr<Task> &task) = 0;

  /**
   * Get remaining work for load balancing
   * @return Amount of work remaining (0 = idle)
   */
  HSHM_GPU_FUN virtual u64 GetWorkRemaining() const { return 0; }
};

}  // namespace gpu
}  // namespace chi

/**
 * CHI_TASK_GPU_CC macro - Generates GPU container allocation/construction kernels
 *
 * Defines:
 * - Device kernel to allocate container via placement new
 * - Device kernel to allocate + Init container
 * - Host functions callable from CPU to create GPU containers
 *
 * @param T Fully-qualified GPU container class name
 */
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#define CHI_TASK_GPU_CC(T)                                                     \
  __global__ void _chimod_gpu_alloc_kernel(T **out) {                          \
    *out = new T();                                                            \
  }                                                                            \
                                                                               \
  __global__ void _chimod_gpu_new_kernel(T **out, const chi::PoolId *pid,      \
                                          chi::u32 cid) {                      \
    T *obj = new T();                                                          \
    obj->Init(*pid, cid);                                                      \
    *out = obj;                                                                \
  }                                                                            \
                                                                               \
  extern "C" void *alloc_chimod_gpu() {                                        \
    void *stream = hshm::GpuApi::CreateStream();                               \
    T **d_out = hshm::GpuApi::Malloc<T *>(sizeof(T *));                        \
    _chimod_gpu_alloc_kernel<<<1, 1, 0,                                        \
        static_cast<cudaStream_t>(stream)>>>(d_out);                           \
    hshm::GpuApi::Synchronize(stream);                                         \
    T *h_ptr = nullptr;                                                        \
    hshm::GpuApi::Memcpy(&h_ptr, d_out, sizeof(T *));                         \
    hshm::GpuApi::Free(d_out);                                                \
    hshm::GpuApi::DestroyStream(stream);                                       \
    return static_cast<void *>(h_ptr);                                         \
  }                                                                            \
                                                                               \
  extern "C" void *new_chimod_gpu(const chi::PoolId *pool_id, chi::u32 cid) { \
    void *stream = hshm::GpuApi::CreateStream();                               \
    T **d_out = hshm::GpuApi::Malloc<T *>(sizeof(T *));                        \
    chi::PoolId *d_pid =                                                       \
        hshm::GpuApi::Malloc<chi::PoolId>(sizeof(chi::PoolId));                \
    hshm::GpuApi::Memcpy(d_pid, pool_id, sizeof(chi::PoolId));                \
    _chimod_gpu_new_kernel<<<1, 1, 0,                                          \
        static_cast<cudaStream_t>(stream)>>>(d_out, d_pid, cid);              \
    hshm::GpuApi::Synchronize(stream);                                         \
    T *h_ptr = nullptr;                                                        \
    hshm::GpuApi::Memcpy(&h_ptr, d_out, sizeof(T *));                         \
    hshm::GpuApi::Free(d_out);                                                \
    hshm::GpuApi::Free(d_pid);                                                \
    hshm::GpuApi::DestroyStream(stream);                                       \
    return static_cast<void *>(h_ptr);                                         \
  }

#else
#define CHI_TASK_GPU_CC(T)
#endif

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
