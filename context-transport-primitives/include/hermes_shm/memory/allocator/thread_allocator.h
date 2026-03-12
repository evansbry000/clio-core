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

#ifndef HSHM_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/thread/lock/mutex.h"

namespace hshm::ipc {

class _ThreadAllocator;
typedef BaseAllocator<_ThreadAllocator> ThreadAllocator;

/**
 * Per-thread allocation partition.
 *
 * Each thread (CPU thread or GPU block) gets its own TaThreadBlock with
 * a private BuddyAllocator. The BuddyAllocator MUST BE LAST because it
 * is followed by its variable-size managed region.
 */
struct TaThreadBlock {
  hipc::atomic<int> initialized_;  /**< 0=uninitialized, 1=ready */
  BuddyAllocator alloc_;          /**< Private buddy allocator (MUST BE LAST) */

  HSHM_CROSS_FUN
  TaThreadBlock() : initialized_(0) {}

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackend &backend, size_t region_size) {
    size_t alloc_region_size = region_size - sizeof(TaThreadBlock);
    alloc_.shm_init(backend, alloc_region_size);
    initialized_ = 1;
    return true;
  }
};

/**
 * Thread allocator with per-thread BuddyAllocator partitions.
 *
 * Designed for single-process, multi-threaded (or multi-GPU-block) use.
 * Each thread/block gets its own BuddyAllocator partition, lazily initialized
 * on first use. A global BuddyAllocator serves as the backing store from
 * which thread partitions are carved, and as a fallback when a thread
 * partition is exhausted.
 *
 * Thread ID mapping:
 *   CPU: caller-provided tid (e.g., thread_local counter)
 *   GPU: blockIdx.x % max_threads_
 *
 * Memory layout (within the backend region):
 *   [_ThreadAllocator header]
 *   [OffsetPtr<TaThreadBlock>[max_threads_] table]   (allocated from global)
 *   [BuddyAllocator global heap ...]
 *   [TaThreadBlock partitions lazily carved from global heap ...]
 */
class _ThreadAllocator : public Allocator {
 public:
  int max_threads_;               /**< Fixed thread count (set at init) */
  size_t thread_unit_;            /**< Bytes per thread partition */
  hshm::Mutex lock_;              /**< Protects global allocator + lazy init */
  OffsetPtr<> thread_table_off_;  /**< Offset to OffsetPtr<TaThreadBlock>[max_threads_] */
  BuddyAllocator alloc_;         /**< Global fallback allocator (MUST BE LAST) */

 public:
  HSHM_CROSS_FUN
  _ThreadAllocator()
      : max_threads_(0), thread_unit_(0) {}

  /**
   * Initialize the thread allocator.
   *
   * @param backend Memory backend
   * @param region_size Size of region (0 = entire backend)
   * @param max_threads Maximum number of threads/GPU blocks
   * @param thread_unit Bytes per thread partition
   * @return true on success
   */
  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackend &backend,
                size_t region_size = 0,
                int max_threads = 32,
                size_t thread_unit = 1024 * 1024) {
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    SetBackend(backend);
    alloc_header_size_ = sizeof(_ThreadAllocator);
    data_start_ = sizeof(_ThreadAllocator);
    region_size_ = region_size;
    max_threads_ = max_threads;
    thread_unit_ = thread_unit;
    lock_.Init();

    // Initialize global buddy allocator over remaining space
    size_t available_size = region_size - sizeof(_ThreadAllocator);
    alloc_.shm_init(backend, available_size);

    // Allocate thread table from global allocator
    size_t table_size = sizeof(OffsetPtr<TaThreadBlock>) * max_threads_;
    OffsetPtr<> table_off = alloc_.AllocateOffset(table_size);
    if (table_off.IsNull()) {
      return false;
    }
    thread_table_off_ = table_off;

    // Null-init all table entries
    auto *table = reinterpret_cast<OffsetPtr<TaThreadBlock>*>(
        GetBackendData() + table_off.load());
    for (int i = 0; i < max_threads_; ++i) {
      table[i] = OffsetPtr<TaThreadBlock>::GetNull();
    }

    return true;
  }

  /**
   * Get the thread table pointer array.
   */
  HSHM_INLINE_CROSS_FUN
  OffsetPtr<TaThreadBlock>* GetThreadTable() {
    return reinterpret_cast<OffsetPtr<TaThreadBlock>*>(
        GetBackendData() + thread_table_off_.load());
  }

  /**
   * Get a TaThreadBlock by tid.
   */
  HSHM_INLINE_CROSS_FUN
  TaThreadBlock* GetThreadBlock(int tid) {
    auto *table = GetThreadTable();
    if (table[tid].IsNull()) {
      return nullptr;
    }
    return reinterpret_cast<TaThreadBlock*>(
        GetBackendData() + table[tid].load());
  }

  /**
   * Lazily initialize thread partition for the given tid.
   * Thread-safe via double-checked locking with mutex.
   */
  HSHM_CROSS_FUN
  bool LazyInitThread(int tid) {
    if (tid < 0 || tid >= max_threads_) {
      return false;
    }

    // Fast path: already initialized
    TaThreadBlock *block = GetThreadBlock(tid);
    if (block != nullptr && static_cast<int>(block->initialized_) == 1) {
      return true;
    }

    // Slow path: lock and initialize
    ScopedMutex scoped_lock(lock_, 0);

    // Double-check after acquiring lock
    block = GetThreadBlock(tid);
    if (block != nullptr && static_cast<int>(block->initialized_) == 1) {
      return true;
    }

    // Allocate a region for the TaThreadBlock + its BuddyAllocator data
    FullPtr<TaThreadBlock> tblock_ptr =
        alloc_.AllocateRegion<TaThreadBlock>(thread_unit_);
    if (tblock_ptr.IsNull()) {
      return false;
    }

    // Initialize the thread block's BuddyAllocator
    MemoryBackend backend = GetBackend();
    tblock_ptr.ptr_->shm_init(backend, thread_unit_);

    // Store in table
    auto *table = GetThreadTable();
    table[tid] = OffsetPtr<TaThreadBlock>(tblock_ptr.shm_.off_.load());

#ifndef HSHM_IS_HOST
    __threadfence();
#endif

    return true;
  }

  /**
   * Get the auto-detected thread ID.
   * GPU: blockIdx.x % max_threads_
   * CPU: 0 (caller should provide explicit tid for multi-threaded use)
   */
  HSHM_INLINE_CROSS_FUN
  int GetAutoTid() {
#if HSHM_IS_GPU
    return static_cast<int>(blockIdx.x) % max_threads_;
#else
    return 0;
#endif
  }

  /**
   * Allocate memory with auto-detected tid.
   * On GPU, automatically uses blockIdx.x % max_threads_.
   * On CPU, defaults to tid=0.
   */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size) {
    return AllocateOffset(size, GetAutoTid());
  }

  /**
   * Allocate memory with explicit tid.
   *
   * @param size Size in bytes
   * @param tid Thread/block ID
   * @return Offset pointer to allocated memory
   */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size, int tid) {
    if (!LazyInitThread(tid)) {
      return OffsetPtr<>::GetNull();
    }

    // Try thread-local allocator first (lock-free for single writer)
    TaThreadBlock *block = GetThreadBlock(tid);
    if (block != nullptr) {
      OffsetPtr<> ptr = block->alloc_.AllocateOffset(size);
      if (!ptr.IsNull()) {
        return ptr;
      }
    }

    // Fallback: allocate from global allocator (with lock)
    ScopedMutex scoped_lock(lock_, 0);
    return alloc_.AllocateOffset(size);
  }


  /**
   * Reallocate memory (not supported — allocate new + copy manually).
   */
  HSHM_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p;
    (void)new_size;
    return OffsetPtr<>::GetNull();
  }

  /**
   * Free memory.
   *
   * Determines which thread block owns the offset by address range,
   * and frees to that allocator. Falls back to global if not found.
   */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    char *base = GetBackendData();
    char *ptr_addr = base + p.load();

    // Check each thread block's range
    auto *table = GetThreadTable();
    for (int i = 0; i < max_threads_; ++i) {
      if (table[i].IsNull()) continue;
      TaThreadBlock *block = reinterpret_cast<TaThreadBlock*>(
          base + table[i].load());
      if (static_cast<int>(block->initialized_) != 1) continue;
      if (block->alloc_.ContainsPtr(ptr_addr)) {
        block->alloc_.FreeOffsetNoNullCheck(p);
        return;
      }
    }

    // Not in any thread block — free to global
    ScopedMutex scoped_lock(lock_, 0);
    alloc_.FreeOffsetNoNullCheck(p);
  }

  /**
   * Free memory with explicit offset (calls FreeOffsetNoNullCheck).
   */
  HSHM_CROSS_FUN
  void FreeOffset(OffsetPtr<> p) {
    if (p.IsNull()) return;
    FreeOffsetNoNullCheck(p);
  }

  /** No-op TLS management (thread IDs are caller-provided) */
  HSHM_CROSS_FUN void CreateTls() {}
  HSHM_CROSS_FUN void FreeTls() {}

  /** Get shared header (not used) */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetSharedHeader() {
    return nullptr;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_
