/**
 * workload_gray_scott.cc — Gray-Scott stencil simulation for CTE GPU bench
 *
 * CTE mode: Stencil compute in HBM, checkpoints offloaded via
 *   GPU-initiated AsyncPutBlob through the CTE runtime.
 * HBM mode: Stencil + cudaMemcpy checkpoint (no CTE).
 * Direct mode: All data in pinned DRAM.
 */

// GPU kernels (visible in both host and device passes)
#include <cstdint>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

struct GSParams { float Du=0.05f, Dv=0.1f, F=0.04f, k=0.06075f, dt=0.2f; };

__device__ inline uint32_t gs_idx(int x, int y, int z, int L) {
  return (uint32_t)(((x+L)%L) + ((y+L)%L)*L + ((z+L)%L)*L*L);
}

__global__ void gs_stencil_hbm(const float *u, const float *v,
                                float *u2, float *v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;
  float lu = u[gs_idx(x-1,y,z,L)]+u[gs_idx(x+1,y,z,L)]
           +u[gs_idx(x,y-1,z,L)]+u[gs_idx(x,y+1,z,L)]
           +u[gs_idx(x,y,z-1,L)]+u[gs_idx(x,y,z+1,L)]-6.0f*u[tid];
  lu/=6.0f;
  float lv = v[gs_idx(x-1,y,z,L)]+v[gs_idx(x+1,y,z,L)]
           +v[gs_idx(x,y-1,z,L)]+v[gs_idx(x,y+1,z,L)]
           +v[gs_idx(x,y,z-1,L)]+v[gs_idx(x,y,z+1,L)]-6.0f*v[tid];
  lv/=6.0f;
  float uv=u[tid], vv=v[tid], uvv=uv*vv*vv;
  u2[tid]=uv+p.dt*(p.Du*lu-uvv+p.F*(1.0f-uv));
  v2[tid]=vv+p.dt*(p.Dv*lv+uvv-(p.F+p.k)*vv);
}

__global__ void gs_stencil_direct(const float *h_u, const float *h_v,
                                   float *h_u2, float *h_v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;
  float lu = h_u[gs_idx(x-1,y,z,L)]+h_u[gs_idx(x+1,y,z,L)]
           +h_u[gs_idx(x,y-1,z,L)]+h_u[gs_idx(x,y+1,z,L)]
           +h_u[gs_idx(x,y,z-1,L)]+h_u[gs_idx(x,y,z+1,L)]-6.0f*h_u[tid];
  lu/=6.0f;
  float lv = h_v[gs_idx(x-1,y,z,L)]+h_v[gs_idx(x+1,y,z,L)]
           +h_v[gs_idx(x,y-1,z,L)]+h_v[gs_idx(x,y+1,z,L)]
           +h_v[gs_idx(x,y,z-1,L)]+h_v[gs_idx(x,y,z+1,L)]-6.0f*h_v[tid];
  lv/=6.0f;
  float uv=h_u[tid], vv=h_v[tid], uvv=uv*vv*vv;
  h_u2[tid]=uv+p.dt*(p.Du*lu-uvv+p.F*(1.0f-uv));
  h_v2[tid]=vv+p.dt*(p.Dv*lv+uvv-(p.F+p.k)*vv);
}

/**
 * CTE checkpoint kernel: GPU-initiated PutBlob.
 * Warp 0 submits AsyncPutBlob for the field data.
 */
__global__ void gs_cte_ckpt_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes,
    chi::u32 step_num,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();

  if (warp_id == 0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);

    // Build blob name: "gs_step_<N>"
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
    char name[32];
    int pos = 0;
    const char *pfx = "gs_";
    while (*pfx) name[pos++] = *pfx++;
    pos += StrT::NumberToStr(name + pos, 32 - pos, step_num);
    name[pos] = '\0';

    hipc::ShmPtr<> blob_shm;
    blob_shm.alloc_id_ = data_alloc_id;
    blob_shm.off_.exchange(data_ptr.shm_.off_.load());

    auto future = cte_client.AsyncPutBlob(
        tag_id, name,
        (chi::u64)0, data_bytes,
        blob_shm, -1.0f,
        wrp_cte::core::Context(), (chi::u32)0,
        chi::PoolQuery::Local());

    if (!future.GetFutureShmPtr().IsNull()) {
      future.Wait();
    }

    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// Alloc kernel for CTE data backend (must be in device pass)
__global__ void gs_cte_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#include <vector>
#include <cstring>

static void gs_init(float *u, float *v, int L) {
  uint32_t total = L*L*L;
  for (uint32_t i = 0; i < total; i++) { u[i]=1.0f; v[i]=0.0f; }
  int lo=L/4, hi=3*L/4;
  for (int z=lo;z<hi;z++) for (int y=lo;y<hi;y++) for (int x=lo;x<hi;x++) {
    uint32_t idx = x+y*L+z*L*L; u[idx]=0.75f; v[idx]=0.25f;
  }
}

static bool gs_poll(int *d_done, int expected, int timeout_sec) {
  int64_t elapsed = 0, timeout_us = (int64_t)timeout_sec * 1000000;
  while (__atomic_load_n(d_done, __ATOMIC_ACQUIRE) < expected &&
         elapsed < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed += 100;
  }
  return __atomic_load_n(d_done, __ATOMIC_ACQUIRE) >= expected;
}

int run_workload_gray_scott(const WorkloadConfig &cfg, const char *mode,
                            WorkloadResult *result) {
  int L = cfg.param_grid_size;
  int steps = cfg.param_steps;
  int ckpt = cfg.param_checkpoint_freq;
  uint32_t total = L*L*L;
  size_t fb = total * sizeof(float);
  GSParams params;
  int threads = 256, blocks = (total+threads-1)/threads;
  std::string m(mode);

  if (m == "cte") {
    // ---- CTE mode: stencil in HBM, checkpoints via AsyncPutBlob ----
    float *d_u,*d_v,*d_u2,*d_v2;
    cudaMalloc(&d_u,fb); cudaMalloc(&d_v,fb);
    cudaMalloc(&d_u2,fb); cudaMalloc(&d_v2,fb);
    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(),hv.data(),L);
    cudaMemcpy(d_u,hu.data(),fb,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,hv.data(),fb,cudaMemcpyHostToDevice);

    // Set up CTE backends for PutBlob
    CHI_IPC->PauseGpuOrchestrator();

    hipc::MemoryBackendId data_id(200, 0);
    hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, fb * 2 + 4 * 1024 * 1024, "", 0);

    hipc::MemoryBackendId scratch_id(201, 0);
    hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, 1 * 1024 * 1024, "", 0);

    hipc::MemoryBackendId heap_id(202, 0);
    hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, 1 * 1024 * 1024, "", 0);

    hipc::FullPtr<char> *d_ptr;
    cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gs_cte_alloc_kernel<<<1, 1>>>(
        static_cast<hipc::MemoryBackend &>(data_backend),
        fb * 2, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError, "CTE alloc kernel failed");
      cudaFreeHost(d_ptr);
      cudaFree(d_u);cudaFree(d_v);cudaFree(d_u2);cudaFree(d_v2);
      CHI_IPC->ResumeGpuOrchestrator();
      return -2;
    }
    hipc::FullPtr<char> array_ptr = *d_ptr;
    cudaFreeHost(d_ptr);

    hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_,
                                   data_backend.data_capacity_);

    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;

    int *d_done;
    cudaMallocHost(&d_done, sizeof(int));

    if (scratch_backend.data_)
      cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    if (heap_backend.data_)
      cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < steps; s++) {
      // Stencil compute (paused orchestrator is fine — stencil doesn't use CTE)
      gs_stencil_hbm<<<blocks,threads>>>(d_u,d_v,d_u2,d_v2,L,params);
      std::swap(d_u,d_u2); std::swap(d_v,d_v2);

      // CTE checkpoint
      if (ckpt > 0 && (s + 1) % ckpt == 0) {
        cudaDeviceSynchronize();

        // Copy field data into the CTE data backend
        cudaMemcpy(array_ptr.ptr_, d_u, fb, cudaMemcpyDeviceToDevice);
        cudaMemcpy(array_ptr.ptr_ + fb, d_v, fb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // Launch CTE checkpoint kernel
        *d_done = 0;
        gs_cte_ckpt_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
            gpu_info, cfg.cte_pool_id, cfg.tag_id,
            cfg.rt_blocks, array_ptr, data_alloc_id,
            fb * 2, (uint32_t)(s + 1), d_done);

        CHI_IPC->ResumeGpuOrchestrator();
        auto *orch = static_cast<chi::gpu::WorkOrchestrator *>(CHI_IPC->gpu_orchestrator_);
        auto *ctrl = orch ? orch->control_ : nullptr;
        if (ctrl) {
          int wait_ms = 0;
          while (ctrl->running_flag == 0 && wait_ms < 5000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++wait_ms;
          }
        }
        int64_t timeout_us = (int64_t)cfg.timeout_sec * 1000000;
        int64_t elapsed_us = 0;
        while (__atomic_load_n(d_done, __ATOMIC_ACQUIRE) < 1 &&
               elapsed_us < timeout_us) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          elapsed_us += 100;
        }
        bool ok = __atomic_load_n(d_done, __ATOMIC_ACQUIRE) >= 1;
        if (!ok) {
          HLOG(kError, "CTE checkpoint timed out at step {}", s + 1);
          break;
        }
        // Pause again for next stencil batch
        CHI_IPC->PauseGpuOrchestrator();
      }
    }
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";

    cudaFreeHost(d_done);
    cudaFree(d_u);cudaFree(d_v);cudaFree(d_u2);cudaFree(d_v2);
    return 0;

  } else if (m == "hbm") {
    // ---- HBM mode: stencil + cudaMemcpy checkpoint ----
    float *d_u,*d_v,*d_u2,*d_v2;
    cudaMalloc(&d_u,fb); cudaMalloc(&d_v,fb);
    cudaMalloc(&d_u2,fb); cudaMalloc(&d_v2,fb);
    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(),hv.data(),L);
    cudaMemcpy(d_u,hu.data(),fb,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,hv.data(),fb,cudaMemcpyHostToDevice);
    float *h_ckpt; cudaMallocHost(&h_ckpt, fb*2);

    auto t0=std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) {
      gs_stencil_hbm<<<blocks,threads>>>(d_u,d_v,d_u2,d_v2,L,params);
      std::swap(d_u,d_u2); std::swap(d_v,d_v2);
      if (ckpt>0 && (s+1)%ckpt==0) {
        cudaMemcpy(h_ckpt,d_u,fb,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ckpt+total,d_v,fb,cudaMemcpyDeviceToHost);
      }
    }
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps=(double)total*sizeof(float)*(2*7+2);
    result->bandwidth_gbps=(bps*steps/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=result->elapsed_ms/steps;
    result->metric_name="ms/step";
    cudaFree(d_u);cudaFree(d_v);cudaFree(d_u2);cudaFree(d_v2);cudaFreeHost(h_ckpt);

  } else if (m == "direct") {
    // ---- Direct DRAM mode ----
    float *h_u,*h_v,*h_u2,*h_v2;
    cudaMallocHost(&h_u,fb);cudaMallocHost(&h_v,fb);
    cudaMallocHost(&h_u2,fb);cudaMallocHost(&h_v2,fb);
    gs_init(h_u,h_v,L);
    auto t0=std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) {
      gs_stencil_direct<<<blocks,threads>>>(h_u,h_v,h_u2,h_v2,L,params);
      cudaDeviceSynchronize();
      std::swap(h_u,h_u2); std::swap(h_v,h_v2);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps=(double)total*sizeof(float)*(2*7+2);
    result->bandwidth_gbps=(bps*steps/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=result->elapsed_ms/steps;
    result->metric_name="ms/step";
    cudaFreeHost(h_u);cudaFreeHost(h_v);cudaFreeHost(h_u2);cudaFreeHost(h_v2);
  } else {
    HLOG(kError, "gray_scott: unknown mode '{}'", mode);
    return -1;
  }
  return 0;
}

#endif  // HSHM_IS_HOST
