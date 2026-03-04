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
 * GPU runtime for MOD_NAME ChiMod
 *
 * Handles GpuSubmitTask execution on the GPU.
 * The megakernel dispatches tasks to this container via gpu::Worker.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/gpu_container.h"
#include "chimaera/MOD_NAME/MOD_NAME_tasks.h"
#include "chimaera/MOD_NAME/autogen/MOD_NAME_methods.h"
#include "chimaera/ipc_manager.h"

namespace chimaera::MOD_NAME {

/**
 * GPU-side container for the MOD_NAME module.
 * Processes GpuSubmitTask on the GPU megakernel.
 */
class GpuRuntime : public chi::gpu::Container {
 public:
  HSHM_GPU_FUN GpuRuntime() = default;
  HSHM_GPU_FUN ~GpuRuntime() override = default;

  /**
   * Execute a task method on the GPU.
   * Dispatched by gpu::Worker after deserialization.
   */
  HSHM_GPU_FUN void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                         chi::gpu::GpuRunContext &rctx) override {
    (void)rctx;
    if (method == Method::kGpuSubmit) {
      auto task = task_ptr.template Cast<GpuSubmitTask>();
      task->result_value_ = (task->test_value_ * 2) + task->gpu_id_;
    }
  }

  /**
   * Allocate and deserialize a task from a local archive.
   * Called by gpu::Worker after ShmTransport::Recv populates the archive.
   */
  HSHM_GPU_FUN hipc::FullPtr<chi::Task> LocalAllocLoadTask(
      chi::u32 method, chi::LocalLoadTaskArchive &archive) override {
    if (method == Method::kGpuSubmit) {
      auto *alloc = gpu_alloc_;
      auto task = alloc->template AllocateObjs<GpuSubmitTask>(1);
      if (task.IsNull()) {
        return hipc::FullPtr<chi::Task>::GetNull();
      }
      new (task.ptr_) GpuSubmitTask();
      archive.SetMsgType(chi::LocalMsgType::kSerializeIn);
      task.ptr_->SerializeIn(archive);
      return task.template Cast<chi::Task>();
    }
    return hipc::FullPtr<chi::Task>::GetNull();
  }

  /**
   * Serialize task output into a local archive.
   * Called by gpu::Worker before ShmTransport::Send writes to the ring buffer.
   */
  HSHM_GPU_FUN void LocalSaveTask(
      chi::u32 method, chi::LocalSaveTaskArchive &archive,
      const hipc::FullPtr<chi::Task> &task) override {
    if (method == Method::kGpuSubmit) {
      auto gpu_task = task.template Cast<GpuSubmitTask>();
      gpu_task->SerializeOut(archive);
    }
  }

  HSHM_GPU_FUN chi::u64 GetWorkRemaining() const override { return 0; }
};

}  // namespace chimaera::MOD_NAME

CHI_TASK_GPU_CC(chimaera::MOD_NAME::GpuRuntime)

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
