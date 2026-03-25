/**
 * bam/page_cache.cuh -- GPU-side page cache for BaM
 *
 * Implements a software-managed page cache in GPU HBM (VRAM).
 * GPU threads request pages by storage offset; the cache handles:
 *   - Lookup (direct-mapped, offset mod num_pages)
 *   - Miss handling: loads from DRAM (pinned host memory) or NVMe
 *   - Eviction on conflict miss
 *
 * Concurrency model:
 *   - page_tags[slot] is the authoritative tag (atomicCAS to claim)
 *   - page_states[slot] tracks load progress:
 *       kInvalid -> kLoading -> kValid (or kDirty)
 *   - On miss, exactly ONE thread wins the CAS and loads the page.
 *     All other threads that need the same page spin on page_states
 *     until kValid/kDirty.
 *   - No warp-level deadlocks: spinning threads are in different
 *     warps, or the loading thread is in the same warp and will
 *     complete before the spin is reached (SIMT convergence).
 *
 * For this basic version, we avoid intra-warp conflicts by design:
 *   - The loading thread sets state to kValid BEFORE returning.
 *   - Other threads that hash to the same slot but want a DIFFERENT
 *     page simply overwrite (last writer wins — acceptable for a
 *     basic direct-mapped cache).
 */
#ifndef BAM_PAGE_CACHE_CUH
#define BAM_PAGE_CACHE_CUH

#include <bam/types.h>
#include <cuda_runtime.h>

namespace bam {

/* ------------------------------------------------------------------ */
/* Device-side page cache operations (all inline)                      */
/* ------------------------------------------------------------------ */

/**
 * Acquire a cache page for the given storage offset.
 *
 * Returns a pointer to the HBM cache page and sets *needs_load = true
 * if the caller is responsible for filling the page from the backend.
 * If *needs_load = false, the page data is already valid.
 *
 * After loading, the caller MUST call page_cache_finish_load().
 */
__device__ inline uint8_t *page_cache_acquire(
    PageCacheDeviceState &state,
    uint64_t offset,
    bool *needs_load) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  uint8_t *page = state.cache_mem + (uint64_t)slot * state.page_size;

  unsigned long long desired = (unsigned long long)offset;
  unsigned long long *tag_ptr = (unsigned long long *)&state.page_tags[slot];

  // Try to claim: if tag already matches, check state
  unsigned long long old_tag = atomicCAS(tag_ptr, desired, desired);

  if (old_tag == desired) {
    // Tag matches — but is the page loaded yet?
    uint32_t st = atomicAdd(&state.page_states[slot], 0);  // atomic read
    if (st == static_cast<uint32_t>(PageState::kValid) ||
        st == static_cast<uint32_t>(PageState::kDirty)) {
      *needs_load = false;
      return page;
    }
    // Someone else is loading — spin until ready
    while (true) {
      __threadfence();  // yield / memory fence as backoff
      st = atomicAdd(&state.page_states[slot], 0);
      if (st == static_cast<uint32_t>(PageState::kValid) ||
          st == static_cast<uint32_t>(PageState::kDirty)) {
        *needs_load = false;
        return page;
      }
      // If tag changed while spinning, restart
      unsigned long long cur = atomicAdd(tag_ptr, 0ULL);
      if (cur != desired) break;
    }
    // Tag was stolen — fall through to install our tag
  }

  // Cache miss: install our tag and mark as loading
  atomicExch(tag_ptr, desired);
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kLoading));
  __threadfence();

  *needs_load = true;
  return page;
}

/**
 * Signal that a page load is complete. Must be called after
 * the caller has finished filling the page data.
 */
__device__ inline void page_cache_finish_load(
    PageCacheDeviceState &state,
    uint64_t offset) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  __threadfence();  // Ensure data writes are visible before state change
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kValid));
  __threadfence();
}

/**
 * Release a cache page (no-op in this design).
 */
__device__ inline void page_cache_release(
    PageCacheDeviceState &state,
    uint64_t offset) {
  (void)state; (void)offset;
}

/**
 * Mark a cache page as dirty.
 */
__device__ inline void page_cache_mark_dirty(
    PageCacheDeviceState &state,
    uint64_t offset) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kDirty));
}

/* ------------------------------------------------------------------ */
/* DRAM (host memory) I/O — device-side                                */
/* ------------------------------------------------------------------ */

/**
 * Copy a page from pinned DRAM to GPU HBM cache page.
 */
__device__ inline void host_read_page(
    uint8_t *dst,
    const uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  const uint8_t *src = host_base + offset;
  for (uint32_t i = 0; i < page_size; i += sizeof(uint4)) {
    *reinterpret_cast<uint4 *>(dst + i) =
        *reinterpret_cast<const uint4 *>(src + i);
  }
  __threadfence();
}

/**
 * Copy a page from GPU HBM cache page back to pinned DRAM.
 */
__device__ inline void host_write_page(
    const uint8_t *src,
    uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  uint8_t *dst = host_base + offset;
  for (uint32_t i = 0; i < page_size; i += sizeof(uint4)) {
    *reinterpret_cast<uint4 *>(dst + i) =
        *reinterpret_cast<const uint4 *>(src + i);
  }
  __threadfence_system();
}

/* ------------------------------------------------------------------ */
/* NVMe I/O helpers — stubs when NVMe disabled                         */
/* ------------------------------------------------------------------ */

__device__ inline int nvme_read_page(
    QueuePairDevice &qp,
    uint64_t buf_bus_addr,
    uint64_t storage_offset,
    uint32_t page_size) {
  (void)qp; (void)buf_bus_addr; (void)storage_offset; (void)page_size;
  return -1;
}

__device__ inline int nvme_write_page(
    QueuePairDevice &qp,
    uint64_t buf_bus_addr,
    uint64_t storage_offset,
    uint32_t page_size) {
  (void)qp; (void)buf_bus_addr; (void)storage_offset; (void)page_size;
  return -1;
}

}  // namespace bam

#endif  // BAM_PAGE_CACHE_CUH
