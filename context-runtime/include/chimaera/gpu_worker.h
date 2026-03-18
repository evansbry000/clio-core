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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_

#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_work_orchestrator.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

/**
 * GPU-side worker that mirrors the CPU Worker API.
 *
 * Runs on block 0, thread 0 of the GPU work orchestrator. Polls both the
 * CPU→GPU queue (cpu2gpu_queue) and the GPU→GPU queue (gpu2gpu_queue)
 * for incoming tasks, deserializes inputs, dispatches to GPU containers,
 * serializes outputs, and signals completion.
 *
 * No STL, no TLS — all HSHM_GPU_FUN.  Container::Run() returns
 * chi::gpu::TaskResume (a C++20 coroutine) enabling cooperative yielding.
 */
class Worker {
 public:
  u32 worker_id_;                    /**< Worker identity */
  u32 lane_id_;                      /**< Lane this worker polls */
  volatile bool is_running_;         /**< Running flag for the poll loop */
  TaskQueue *cpu2gpu_queue_;         /**< CPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *gpu2gpu_queue_;         /**< GPU → GPU queue (GPU work orchestrator polls) */
  PoolManager *pool_mgr_;            /**< GPU-side container lookup table */
  char *queue_backend_base_;         /**< Base of queue backend for ShmPtr resolution */
  RunContext rctx_;               /**< Reused run context per task */
  WorkOrchestratorControl *dbg_ctrl_; /**< Debug control struct (pinned, CPU-readable) */

  static constexpr u32 kMaxSuspended = 16;

  /** State for a coroutine that yielded mid-execution */
  struct SuspendedTask {
    TaskResume coro;               /**< Outer coroutine (e.g., PutBlob) */
    RunContext rctx;               /**< Per-coroutine RunContext */
    FutureShm *fshm;              /**< Deferred completion context */
    Container *container;
    u32 method_id;
    hipc::FullPtr<Task> task_ptr;
    bool is_gpu2gpu;
    bool is_copy_path;             /**< true = DispatchTask, false = DispatchTaskDirect */
    bool occupied;
  };
  SuspendedTask suspended_[kMaxSuspended];
  u32 num_suspended_;

  /**
   * Initialize the worker with queue and pool manager pointers.
   *
   * @param worker_id Logical worker ID
   * @param lane_id Lane index this worker polls from
   * @param cpu2gpu_queue CPU→GPU queue pointer (pinned host memory)
   * @param gpu2gpu_queue GPU→GPU queue pointer (device memory)
   * @param pool_mgr GPU-side pool manager for container lookup
   * @param queue_backend_base Base pointer of queue backend for ShmPtr offsets
   */
  HSHM_GPU_FUN void Init(u32 worker_id,
                          u32 lane_id,
                          TaskQueue *cpu2gpu_queue,
                          TaskQueue *gpu2gpu_queue,
                          PoolManager *pool_mgr,
                          char *queue_backend_base,
                          WorkOrchestratorControl *dbg_ctrl = nullptr) {
    worker_id_ = worker_id;
    lane_id_ = lane_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    dbg_ctrl_ = dbg_ctrl;
    is_running_ = true;
    num_suspended_ = 0;
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      suspended_[i].occupied = false;
    }
#if HSHM_IS_GPU_COMPILER
    rctx_ = RunContext(blockIdx.x, threadIdx.x);
#else
    rctx_ = RunContext(0, 0);
#endif
  }

  /**
   * Process one iteration of the poll loop.
   *
   * Checks both queues for pending work. Returns true if any work was done,
   * false if both queues were empty (caller can decide to spin or yield).
   *
   * @return true if at least one task was processed
   */
  /** Write debug state to pinned memory (CPU-readable) */
  HSHM_GPU_FUN void DbgState([[maybe_unused]] u32 state) {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_last_state[lane_id_] = state;
      dbg_ctrl_->dbg_num_suspended[lane_id_] = num_suspended_;
    }
#endif
  }

  HSHM_GPU_FUN void DbgPoll() {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_poll_count[lane_id_]++;
    }
#endif
  }

  HSHM_GPU_FUN bool PollOnce() {
    bool did_work = false;
    DbgPoll();

    // 1. Resume suspended coroutines first (may free slots)
    if (num_suspended_ > 0) {
      did_work |= ResumeAllSuspended();
    }

    // 2. Accept new tasks only if we have suspended-coroutine capacity.
    if (num_suspended_ < kMaxSuspended) {
      did_work |= ProcessGpu2GpuTask();
      if (lane_id_ == 0) {
        did_work |= ProcessCpu2GpuTask();
      }
    }

#ifndef NDEBUG
    // Periodic diagnostic (every ~1M polls, lane 0 only)
    if (lane_id_ == 0 && dbg_ctrl_) {
      u64 cnt = dbg_ctrl_->dbg_poll_count[0];
      if (cnt > 0 && (cnt & 0xFFFFF) == 0) {
        printf("[GPU Worker 0] poll_count=%llu cpu2gpu=%p gpu2gpu=%p susp=%u\n",
               (unsigned long long)cnt, (void*)cpu2gpu_queue_,
               (void*)gpu2gpu_queue_, (unsigned)num_suspended_);
      }
    }
#endif

    return did_work;
  }

  /**
   * Signal the worker to stop polling.
   */
  HSHM_GPU_FUN void Stop() {
    is_running_ = false;
  }

  /**
   * Finalize and cleanup the worker.
   * Currently a no-op; placeholder for future resource release.
   */
  HSHM_GPU_FUN void Finalize() {
    is_running_ = false;
  }

 private:
  /**
   * Resume all suspended coroutines via their innermost yielded handles.
   * @return true if any work was done
   */
  HSHM_GPU_FUN bool ResumeAllSuspended() {
    bool did_work = false;
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      if (!suspended_[i].occupied) continue;

      auto &s = suspended_[i];

#ifndef NDEBUG
      if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
        dbg_ctrl_->dbg_resume_checks[lane_id_]++;
      }
#endif

      // If suspended on a Future (co_await future), only resume when
      // FUTURE_COMPLETE is set on the awaited FutureShm.
      if (s.rctx.awaited_fshm_) {
        auto *awaited = reinterpret_cast<FutureShm *>(s.rctx.awaited_fshm_);
        if (!awaited->flags_.Any(FutureShm::FUTURE_COMPLETE)) {
          continue;  // Sub-task not done yet, skip
        }
        // Deserialize sub-task output BEFORE resuming the coroutine.
        // This avoids allocation issues inside resumed coroutine frames.
#if !HSHM_IS_HOST
        DeserializeAwaitedOutput(s.rctx, awaited);
#endif
        // Clear awaited pointers before resuming so the coroutine can
        // await another Future without stale state.
        s.rctx.awaited_fshm_ = nullptr;
        s.rctx.awaited_task_ = nullptr;
      }

      // Resume the innermost yielded coroutine stored in per-task rctx
      if (s.rctx.coro_handle_) {
        s.rctx.is_yielded_ = false;
        DbgState(4);  // Resuming
#ifndef NDEBUG
        printf("[GPU Worker %u] Resuming coro_handle_ %p (done=%d)\n",
               lane_id_, s.rctx.coro_handle_.address(),
               (int)s.rctx.coro_handle_.done());
#endif
        s.rctx.coro_handle_.resume();
#ifndef NDEBUG
        printf("[GPU Worker %u] coro_handle_.resume() returned (done=%d)\n",
               lane_id_, (int)s.rctx.coro_handle_.done());
#endif
        did_work = true;

        // Chain-resume: if inner completed, resume its caller.
        ChainResumeCallers(s);
      }

      // Check if the top-level coroutine is now complete
      if (s.coro.done()) {
        DbgState(5);  // Completed
        s.occupied = false;
        --num_suspended_;
        if (s.is_copy_path) {
          SerializeAndComplete(s.fshm, s.container,
                               s.method_id, s.task_ptr,
                               s.is_gpu2gpu);
        } else {
          CompleteAndResumeParent(s.fshm, s.is_gpu2gpu);
        }
      }
    }
    return did_work;
  }

  /**
   * Chain-resume callers after an inner coroutine completes.
   *
   * Because FinalAwaiter returns noop_coroutine() on GPU (no symmetric
   * transfer), the Worker must explicitly detect inner completion and
   * resume the caller. The caller's await_resume safely destroys the
   * inner frame (stack is fully unwound) and may yield again or complete.
   *
   * This loops to handle multi-level nesting (A co_awaits B co_awaits C).
   */
  HSHM_GPU_FUN void ChainResumeCallers(SuspendedTask &s) {
    while (s.rctx.coro_handle_ && s.rctx.coro_handle_.done()) {
      // Inner completed. Get its caller from the promise.
      auto typed = std::coroutine_handle<TaskResume::promise_type>::from_address(
          s.rctx.coro_handle_.address());
      auto caller = typed.promise().caller_handle_;

#ifndef NDEBUG
      printf("[GPU Worker %u] ChainResume: inner %p done, caller %p\n",
             lane_id_, s.rctx.coro_handle_.address(),
             caller ? caller.address() : nullptr);
#endif

      if (!caller) {
        s.rctx.coro_handle_ = nullptr;
        break;
      }

#ifndef NDEBUG
      printf("[GPU Worker %u] ChainResume: resuming caller %p\n",
             lane_id_, caller.address());
#endif
      caller.resume();
#ifndef NDEBUG
      printf("[GPU Worker %u] ChainResume: caller.resume() returned, coro_handle_=%p done=%d\n",
             lane_id_, s.rctx.coro_handle_ ? s.rctx.coro_handle_.address() : nullptr,
             s.rctx.coro_handle_ ? (int)s.rctx.coro_handle_.done() : -1);
#endif
    }
  }

  /**
   * Find a free slot in the suspended_ array.
   * Caller must ensure num_suspended_ < kMaxSuspended before calling.
   */
  HSHM_GPU_FUN u32 FindFreeSlot() {
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      if (!suspended_[i].occupied) return i;
    }
    return 0;  // Should never reach here if caller checks capacity
  }

  /**
   * Save a yielded coroutine into a free suspended slot.
   * Re-points both the outer and inner coroutine promises to the stored RunContext.
   */
  HSHM_GPU_FUN void SuspendCoroutine(TaskResume &&coro, RunContext &task_rctx,
                                       FutureShm *fshm, Container *container,
                                       u32 method_id, hipc::FullPtr<Task> task_ptr,
                                       bool is_gpu2gpu, bool is_copy_path) {
    u32 slot = FindFreeSlot();
    auto &s = suspended_[slot];
    s.coro = static_cast<TaskResume&&>(coro);
    s.rctx = task_rctx;
    // Re-point the outer promise to the stored (non-stack) RunContext
    s.coro.get_handle().promise().set_run_context(&s.rctx);
    // Re-point the inner (yielded) coroutine's promise as well
    if (s.rctx.coro_handle_) {
      auto typed = std::coroutine_handle<TaskResume::promise_type>::from_address(
          s.rctx.coro_handle_.address());
      typed.promise().set_run_context(&s.rctx);
    }
    s.fshm = fshm;
    s.container = container;
    s.method_id = method_id;
    s.task_ptr = task_ptr;
    s.is_gpu2gpu = is_gpu2gpu;
    s.is_copy_path = is_copy_path;
    s.occupied = true;
    ++num_suspended_;
  }

  /**
   * Deserialize output from a completed sub-task's FutureShm ring buffer
   * onto the sub-task object. Called BEFORE resuming a suspended coroutine.
   * Marked __noinline__ to isolate stack usage and prevent compiler
   * from merging this with the coroutine resume code path.
   */
#if !HSHM_IS_HOST
  HSHM_GPU_FUN void DeserializeAwaitedOutput(RunContext &rctx,
                                              FutureShm *awaited) {
    if (!rctx.awaited_task_ || awaited->output_.total_written_.load() == 0) {
      return;
    }
    hipc::threadfence();
    auto *sub_task = reinterpret_cast<Task *>(rctx.awaited_task_);
    Container *sub_container = pool_mgr_->GetContainer(sub_task->pool_id_);
    if (!sub_container) return;

    hshm::lbm::LbmContext ctx;
    ctx.copy_space = awaited->copy_space;
    ctx.shm_info_ = &awaited->output_;
    LocalLoadTaskArchive load_ar(CHI_GPU_HEAP);
    hshm::lbm::ShmTransport::RecvDevice(load_ar, ctx);
    hipc::FullPtr<Task> sub_task_ptr;
    sub_task_ptr.ptr_ = sub_task;
    sub_container->LocalLoadTaskOutput(
        sub_task->method_, load_ar, sub_task_ptr);
  }
#endif

  /**
   * Serialize output and signal completion.
   *
   * If the FutureShm has a parent RunContext (intra-GPU sub-task), the
   * output is deserialized directly into the parent's awaited_task_ and
   * the parent coroutine is resumed immediately (same thread, no polling).
   *
   * If no parent (top-level client task), FUTURE_COMPLETE is set so the
   * client-side Wait()/polling sees it.
   */
  HSHM_GPU_FUN void SerializeAndComplete(FutureShm *fshm, Container *container,
                                           u32 method_id,
                                           hipc::FullPtr<Task> &task_ptr,
                                           bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    // Serialize output into FutureShm ring buffer
    hshm::lbm::LbmContext out_ctx;
    out_ctx.copy_space = fshm->copy_space;
    out_ctx.shm_info_ = &fshm->output_;

    auto *heap = CHI_GPU_HEAP;
    LocalSaveTaskArchive save_ar(LocalMsgType::kSerializeOut, heap);
    container->LocalSaveTask(method_id, save_ar, task_ptr);
    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::SendDevice(save_ar, out_ctx);
    } else {
      hshm::lbm::ShmTransport::Send(save_ar, out_ctx);
    }

    // Destroy task
    task_ptr.ptr_->~Task();

#ifndef NDEBUG
    // Debug: record total_written_ after serialization
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_ser_total_written[lane_id_] =
          fshm->output_.total_written_.load();
      dbg_ctrl_->dbg_ser_method[lane_id_] = method_id;
    }
#endif

    // Fence before signaling to ensure ring buffer data is visible.
    hipc::threadfence();

    // Signal completion
    if (is_gpu2gpu) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }

    // If this sub-task has a parent waiting on it (same worker),
    // deserialize output and resume the parent directly.
    ResumeParentIfPresent(fshm);
#endif
  }

  /**
   * Signal completion for a direct-path (no serialization) task.
   * Checks for parent resumption.
   */
  HSHM_GPU_FUN void CompleteAndResumeParent(FutureShm *fshm,
                                              bool is_gpu2gpu) {
    if (is_gpu2gpu) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }
    ResumeParentIfPresent(fshm);
  }

  /**
   * If the completed sub-task's FutureShm has a parent RunContext,
   * deserialize the sub-task output and resume the parent coroutine.
   *
   * On GPU, parent and child run on the same worker thread, so this
   * is safe without any synchronization.
   */
  HSHM_GPU_FUN void ResumeParentIfPresent(FutureShm *fshm) {
#if !HSHM_IS_HOST
    auto *parent_rctx = reinterpret_cast<RunContext *>(fshm->parent_gpu_rctx_);
    if (!parent_rctx) return;

    // Deserialize sub-task output into the parent's awaited_task
    DeserializeAwaitedOutput(*parent_rctx, fshm);

    // Clear awaited state before resuming
    parent_rctx->awaited_fshm_ = nullptr;
    parent_rctx->awaited_task_ = nullptr;
    parent_rctx->is_yielded_ = false;

    // Resume the inner coroutine (the one that co_awaited the Future)
    if (parent_rctx->coro_handle_) {
#ifndef NDEBUG
      printf("[GPU Worker %u] ResumeParentIfPresent: resuming parent coro %p\n",
             lane_id_, parent_rctx->coro_handle_.address());
#endif
      parent_rctx->coro_handle_.resume();
#ifndef NDEBUG
      printf("[GPU Worker %u] ResumeParentIfPresent: resume returned, done=%d\n",
             lane_id_, (int)parent_rctx->coro_handle_.done());
#endif

      // Chain-resume callers if inner completed (FinalAwaiter returns noop)
      // Find the SuspendedTask that owns this RunContext
      for (u32 i = 0; i < kMaxSuspended; ++i) {
        if (suspended_[i].occupied && &suspended_[i].rctx == parent_rctx) {
          ChainResumeCallers(suspended_[i]);
          break;
        }
      }
    }
#endif
  }

  /**
   * Try to pop and process one task from the given queue.
   *
   * Steps:
   * 1. Pop a Future<Task> from lane (0,0)
   * 2. Resolve FutureShm via queue_backend_base_
   * 3. Look up the target container in pool_mgr_
   * 4. Deserialize input, dispatch Run(), serialize output
   * 5. Mark FUTURE_COMPLETE
   *
   * @param queue Queue to pop from
   * @return true if a task was processed, false if queue was empty
   */
  HSHM_GPU_FUN bool ProcessCpu2GpuTask() {
    return ProcessNewTask(cpu2gpu_queue_, 0, false);
  }

  HSHM_GPU_FUN bool ProcessGpu2GpuTask() {
    return ProcessNewTask(gpu2gpu_queue_, lane_id_, true);
  }

  HSHM_GPU_FUN bool ProcessNewTask(TaskQueue *queue, u32 lane_id,
                                    bool is_gpu2gpu) {
    if (!queue) return false;

    auto &lane = queue->GetLane(lane_id, 0);
    Future<Task> future;
    if (is_gpu2gpu) {
      if (!lane.PopDevice(future)) return false;
    } else {
      if (!lane.Pop(future)) return false;
    }

#ifndef NDEBUG
    printf("[GPU Worker %u] Popped task from %s queue, lane=%u\n",
           lane_id_, is_gpu2gpu ? "gpu2gpu" : "cpu2gpu", (unsigned)lane_id);
#endif

    // Resolve FutureShm: queue-dependent resolution
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) {
#ifndef NDEBUG
      printf("[GPU Worker %u] FutureShm sptr is null\n", lane_id_);
#endif
      return true;
    }
    size_t off = sptr.off_.load();
    FutureShm *fshm;
    if (!is_gpu2gpu) {
      // CPU→GPU (SendToGpu): relative offset from queue backend (pinned host)
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      // GPU→GPU: absolute UVA pointer (device memory or pinned host via UVA)
      fshm = reinterpret_cast<FutureShm *>(off);
    }
    if (!fshm) {
      return true;  // Consumed slot but bad pointer
    }

    // Look up the target container
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
#ifndef NDEBUG
    printf("[GPU Worker %u] Resolving container pool=%u.%u method=%u off=%llu fshm=%p\n",
           lane_id_, (unsigned)pool_id.major_, (unsigned)pool_id.minor_,
           (unsigned)method_id, (unsigned long long)off, (void*)fshm);
#endif
    Container *container = pool_mgr_->GetContainer(pool_id);
    DbgState(1);  // Popped task
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_last_method[lane_id_] = method_id;
    }
#endif
    if (!container) {
#ifndef NDEBUG
      printf("[GPU Worker %u] No container for pool=%u.%u, completing\n",
             lane_id_, (unsigned)pool_id.major_, (unsigned)pool_id.minor_);
#endif
      // No container registered — mark complete with no output
      CompleteAndResumeParent(fshm, is_gpu2gpu);
      return true;
    }

    bool is_copy = fshm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT);
#ifndef NDEBUG
    printf("[GPU Worker %u] Dispatching pool=%u.%u method=%u copy=%d\n",
           lane_id_, (unsigned)pool_id.major_, (unsigned)pool_id.minor_,
           (unsigned)method_id, (int)is_copy);
#endif
    if (is_copy) {
      DispatchTask(fshm, container, method_id, is_gpu2gpu);
    } else {
      DispatchTaskDirect(fshm, container, method_id, is_gpu2gpu);
    }
    return true;
  }

  /**
   * Run a task directly from its pointer without (de)serialization.
   *
   * Used by the SendGpuForward path where the task object is already in
   * GPU-accessible memory. Results are written in-place to the task object;
   * the caller reads them after FUTURE_COMPLETE is observed.
   *
   * @param fshm FutureShm with client_task_vaddr_ set to absolute Task*
   * @param container Target GPU container
   * @param method_id Method to dispatch
   */
  HSHM_GPU_FUN void DispatchTaskDirect(FutureShm *fshm, Container *container,
                                        u32 method_id, bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

    // Set allocator on container for cross-library calls
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    // Reconstruct FullPtr from absolute UVA pointer stored in client_task_vaddr_
    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    // Create a per-task RunContext (copy allocator settings from template)
    RunContext task_rctx = rctx_;

    // Execute the task — results are written into task_ptr in-place.
    {
      TaskResume coro = container->Run(method_id, task_ptr, task_rctx);
      coro.get_handle().promise().set_run_context(&task_rctx);
      coro.resume();

      if (!coro.done()) {
        SuspendCoroutine(static_cast<TaskResume&&>(coro), task_rctx,
                         fshm, container, method_id, task_ptr,
                         is_gpu2gpu, false);
        return;
      }
    }

    // Signal completion (and resume parent if this is a sub-task)
    CompleteAndResumeParent(fshm, is_gpu2gpu);
  }

  /**
   * Deserialize input, run container method, serialize output, and complete.
   *
   * @param fshm FutureShm containing I/O ring buffers
   * @param container Target GPU container
   * @param method_id Method to dispatch
   */
  HSHM_GPU_FUN void DispatchTask(FutureShm *fshm, Container *container,
                                  u32 method_id, bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 10;
    }
    printf("[GPU Worker %u] DispatchTask: method=%u fshm=%p\n",
           lane_id_, (unsigned)method_id, (void*)fshm);

    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 11;
    }
#endif

    // Step 1: Deserialize input from FutureShm ring buffer
    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;

#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 12;
      // Read ring buffer state with volatile reads (no system atomics)
      volatile size_t *tw_ptr = reinterpret_cast<volatile size_t*>(
          &fshm->input_.total_written_);
      volatile size_t *cs_ptr = reinterpret_cast<volatile size_t*>(
          &fshm->input_.copy_space_size_);
      dbg_ctrl_->dbg_input_tw[lane_id_] = *tw_ptr;
      dbg_ctrl_->dbg_input_cs[lane_id_] = *cs_ptr;
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 13;
    }
#endif

    // Set allocator on container for cross-library calls
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    LocalLoadTaskArchive load_ar(CHI_GPU_HEAP);

#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 15;
    }

    // DEBUG: inline ReadTransfer to find exact hang point
    {
      size_t ring_size = in_ctx.shm_info_->copy_space_size_.load_system();
      if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
        dbg_ctrl_->dbg_dispatch_step[lane_id_] = 16;
        dbg_ctrl_->dbg_input_cs[lane_id_] = ring_size;
      }
      size_t total_read = in_ctx.shm_info_->total_read_.load_system();
      if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
        dbg_ctrl_->dbg_dispatch_step[lane_id_] = 17;
      }
      size_t total_written = in_ctx.shm_info_->total_written_.load_system();
      if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
        dbg_ctrl_->dbg_dispatch_step[lane_id_] = 18;
        dbg_ctrl_->dbg_input_tw[lane_id_] = total_written;
      }
      printf("[GPU Worker %u] ReadTransfer probe: ring=%llu read=%llu written=%llu\n",
             lane_id_, (unsigned long long)ring_size,
             (unsigned long long)total_read, (unsigned long long)total_written);
      if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
        dbg_ctrl_->dbg_dispatch_step[lane_id_] = 19;
      }
    }
#endif

    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::RecvDevice(load_ar, in_ctx);
    } else {
      hshm::lbm::ShmTransport::Recv(load_ar, in_ctx);
    }

#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[lane_id_] = 20;
    }
#endif
    load_ar.SetMsgType(LocalMsgType::kSerializeIn);

#ifndef NDEBUG
    printf("[GPU Worker %u] DispatchTask: Recv done, allocating task\n", lane_id_);
#endif

    // Step 2: Allocate and load the task via container
    hipc::FullPtr<Task> task_ptr =
        container->LocalAllocLoadTask(method_id, load_ar);

    if (task_ptr.IsNull()) {
#ifndef NDEBUG
      printf("[GPU Worker %u] DispatchTask: task_ptr is null\n", lane_id_);
#endif
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return;
    }

#ifndef NDEBUG
    printf("[GPU Worker %u] DispatchTask: task allocated, running\n", lane_id_);
#endif

    // Step 3: Execute the task (coroutine: resume + destroy)
    // Create a per-task RunContext (copy allocator settings from template)
    RunContext task_rctx = rctx_;
    {
      TaskResume coro = container->Run(method_id, task_ptr, task_rctx);
      coro.get_handle().promise().set_run_context(&task_rctx);
      coro.resume();

      DbgState(2);  // Dispatched
#ifndef NDEBUG
      printf("[GPU Worker %u] DispatchTask: coro resumed, done=%d\n",
             lane_id_, (int)coro.done());
#endif

      if (!coro.done()) {
        DbgState(3);  // Suspending
        SuspendCoroutine(static_cast<TaskResume&&>(coro), task_rctx,
                         fshm, container, method_id, task_ptr,
                         is_gpu2gpu, true);
        return;
      }
    }

#ifndef NDEBUG
    printf("[GPU Worker %u] DispatchTask: serializing output\n", lane_id_);
#endif

    // Step 4: Serialize output into FutureShm ring buffer
    SerializeAndComplete(fshm, container, method_id, task_ptr,
                         is_gpu2gpu);
#ifndef NDEBUG
    printf("[GPU Worker %u] DispatchTask: done\n", lane_id_);
#endif
#endif  // !HSHM_IS_HOST
  }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
