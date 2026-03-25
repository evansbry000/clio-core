/**
 * workload_gray_scott.cc — Gray-Scott stencil simulation for CTE GPU bench
 */

// GPU kernels
#include <cstdint>

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

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
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

  if (m == "hbm" || m == "cte") {
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
