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
 * GPU runtime for Admin ChiMod
 *
 * Empty GPU container — all admin methods are no-ops on the GPU side.
 * This allows the admin pool to have a GPU container registered,
 * which is required for the megakernel's gpu::PoolManager to route
 * admin tasks (e.g., RegisterGpuContainer) on the GPU.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/gpu_container.h"

namespace chimaera::admin {

class GpuRuntime : public chi::gpu::Container {
 public:
  HSHM_GPU_FUN GpuRuntime() = default;
  HSHM_GPU_FUN ~GpuRuntime() override = default;

  HSHM_GPU_FUN void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                         chi::gpu::GpuRunContext &rctx) override {
    (void)method;
    (void)task_ptr;
    (void)rctx;
    // All admin methods are no-ops on GPU
  }

  /** Admin tasks don't run on GPU — return null */
  HSHM_GPU_FUN hipc::FullPtr<chi::Task> LocalAllocLoadTask(
      chi::u32 method, chi::LocalLoadTaskArchive &archive) override {
    (void)method;
    (void)archive;
    return hipc::FullPtr<chi::Task>::GetNull();
  }

  /** Admin tasks don't run on GPU — no-op */
  HSHM_GPU_FUN void LocalSaveTask(
      chi::u32 method, chi::LocalSaveTaskArchive &archive,
      const hipc::FullPtr<chi::Task> &task) override {
    (void)method;
    (void)archive;
    (void)task;
  }

  HSHM_GPU_FUN chi::u64 GetWorkRemaining() const override { return 0; }
};

}  // namespace chimaera::admin

CHI_TASK_GPU_CC(chimaera::admin::GpuRuntime)

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
