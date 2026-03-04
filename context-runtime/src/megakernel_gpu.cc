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
 * GPU Megakernel implementation
 *
 * A persistent GPU kernel that polls CPU→GPU and GPU→GPU queues for tasks,
 * dispatches them to GPU-side containers via gpu::Worker / gpu::PoolManager,
 * and communicates results back through FutureShm completion flags.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/ipc_manager.h"
#include "chimaera/megakernel.h"
#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_worker.h"
#include "chimaera/config_manager.h"

namespace chi {

/**
 * GPU Megakernel - persistent kernel for GPU task execution.
 *
 * All threads in all blocks initialize per-block IpcManagers (ArenaAllocators).
 * Block 0, thread 0 runs the gpu::Worker poll loop that:
 * 1. Polls to_gpu_queue (CPU→GPU) and gpu_to_gpu_queue (GPU→GPU)
 * 2. Deserializes task inputs via ShmTransport
 * 3. Looks up the target container via gpu::PoolManager
 * 4. Dispatches container->Run()
 * 5. Serializes output and marks FUTURE_COMPLETE
 *
 * @param pool_mgr Device-side pool manager for container lookup
 * @param control Pinned host memory control structure for lifecycle signaling
 * @param gpu_info IPC info with backend, queues, and queue_backend_base
 * @param num_blocks Total number of blocks in the grid
 */
__global__ void chimaera_megakernel(gpu::PoolManager *pool_mgr,
                                     MegakernelControl *control,
                                     IpcManagerGpuInfo gpu_info,
                                     u32 num_blocks) {
  // All threads: initialize per-block IpcManager (ArenaAllocators)
  CHIMAERA_MEGAKERNEL_INIT(gpu_info, num_blocks);

  // Only block 0, thread 0 runs the worker loop
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->running_flag = 1;

    gpu::Worker worker;
    worker.Init(0,
                gpu_info.to_gpu_queue,
                gpu_info.gpu_to_gpu_queue,
                pool_mgr,
                gpu_info.queue_backend_base);

    // Poll until exit signal
    while (!control->exit_flag) {
      worker.PollOnce();
    }

    worker.Finalize();
  }

  // Other blocks/threads: wait for exit signal
  // Future: additional blocks could run their own worker loops
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    while (!control->exit_flag) {
      // Spin-wait
    }
  }
}

//==============================================================================
// MegakernelLauncher out-of-class method implementations
//==============================================================================

/**
 * Launch the persistent megakernel on the GPU.
 *
 * Allocates control structure and PoolManager, then launches the kernel
 * with the provided gpu_info for queue and backend pointers.
 */
bool MegakernelLauncher::Launch(const IpcManagerGpuInfo &gpu_info, u32 blocks,
                                 u32 threads_per_block) {
  if (is_launched_) {
    return true;
  }

  // Allocate MegakernelControl in pinned host memory
  MegakernelControl *pinned_control = nullptr;
  cudaMallocHost(&pinned_control, sizeof(MegakernelControl));
  control_ = pinned_control;
  if (!control_) {
    HLOG(kError, "MegakernelLauncher: Failed to allocate control structure");
    return false;
  }
  control_->exit_flag = 0;
  control_->running_flag = 0;

  // Allocate gpu::PoolManager on device
  gpu::PoolManager *d_pm = hshm::GpuApi::Malloc<gpu::PoolManager>(
      sizeof(gpu::PoolManager));
  if (!d_pm) {
    HLOG(kError, "MegakernelLauncher: Failed to allocate GPU PoolManager");
    cudaFreeHost(control_);
    control_ = nullptr;
    return false;
  }
  d_pool_mgr_ = d_pm;

  // Initialize PoolManager on device (zero-initialize)
  gpu::PoolManager host_pm;  // Default constructor zeros slots
  hshm::GpuApi::Memcpy(d_pm, &host_pm, sizeof(gpu::PoolManager));

  // Increase GPU stack size for deep template call chains
  cudaDeviceSetLimit(cudaLimitStackSize, 131072);

  // Create dedicated stream so the persistent megakernel doesn't block
  // other kernel launches (e.g. GPU container allocation kernels)
  stream_ = hshm::GpuApi::CreateStream();

  // Save launch parameters for Pause/Resume
  blocks_ = blocks;
  threads_per_block_ = threads_per_block;

  // Launch persistent megakernel with queue and backend info
  HLOG(kInfo, "Launching megakernel with {} blocks, {} threads/block",
       blocks, threads_per_block);
  chimaera_megakernel<<<blocks, threads_per_block, 0,
      static_cast<cudaStream_t>(stream_)>>>(
      d_pm, control_, gpu_info, blocks);

  is_launched_ = true;
  HLOG(kInfo, "Megakernel launched successfully");
  return true;
}

/**
 * Stop the megakernel and free resources.
 */
void MegakernelLauncher::Finalize() {
  if (!is_launched_) {
    return;
  }

  // Signal megakernel to exit
  if (control_) {
    control_->exit_flag = 1;
  }

  // Wait for kernel to finish on its dedicated stream
  if (stream_) {
    hshm::GpuApi::Synchronize(stream_);
    hshm::GpuApi::DestroyStream(stream_);
    stream_ = nullptr;
  } else {
    hshm::GpuApi::Synchronize();
  }

  // Free resources
  if (d_pool_mgr_) {
    hshm::GpuApi::Free(d_pool_mgr_);
    d_pool_mgr_ = nullptr;
  }
  if (control_) {
    cudaFreeHost(control_);
    control_ = nullptr;
  }

  is_launched_ = false;
  HLOG(kInfo, "Megakernel finalized");
}

/**
 * Pause the megakernel by signaling exit and waiting for completion.
 * Frees SMs so other kernels (e.g., GPU container allocation) can run.
 * The device-side PoolManager, control structure, and stream are preserved.
 */
void MegakernelLauncher::Pause() {
  if (!is_launched_) {
    return;
  }

  // Signal megakernel to exit
  control_->exit_flag = 1;

  // Wait for kernel to finish on its dedicated stream
  hshm::GpuApi::Synchronize(stream_);

  is_launched_ = false;
  HLOG(kInfo, "Megakernel paused");
}

/**
 * Resume a paused megakernel with the same parameters.
 * Re-launches the persistent kernel on the existing stream.
 */
void MegakernelLauncher::Resume(const IpcManagerGpuInfo &gpu_info) {
  if (is_launched_) {
    return;
  }

  // Reset control flags for new launch
  control_->exit_flag = 0;
  control_->running_flag = 0;

  // Re-launch megakernel on the existing stream
  auto *d_pm = static_cast<gpu::PoolManager *>(d_pool_mgr_);
  chimaera_megakernel<<<blocks_, threads_per_block_, 0,
      static_cast<cudaStream_t>(stream_)>>>(
      d_pm, control_, gpu_info, blocks_);

  is_launched_ = true;
  HLOG(kInfo, "Megakernel resumed with {} blocks, {} threads/block",
       blocks_, threads_per_block_);
}

/**
 * Register a GPU container with the device-side PoolManager.
 *
 * Copies PoolManager state to host, updates the container mapping,
 * then copies back. Safe because the megakernel only reads between tasks.
 */
void MegakernelLauncher::RegisterGpuContainer(const PoolId &pool_id,
                                               void *gpu_container_ptr) {
  if (!d_pool_mgr_ || !gpu_container_ptr) {
    return;
  }
  auto *d_pm = static_cast<gpu::PoolManager *>(d_pool_mgr_);
  gpu::PoolManager host_pm;
  hshm::GpuApi::Memcpy(&host_pm, d_pm, sizeof(gpu::PoolManager));
  host_pm.RegisterContainer(pool_id,
                             static_cast<gpu::Container *>(gpu_container_ptr));
  hshm::GpuApi::Memcpy(d_pm, &host_pm, sizeof(gpu::PoolManager));
  HLOG(kInfo, "Registered GPU container for pool {}", pool_id);
}

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
