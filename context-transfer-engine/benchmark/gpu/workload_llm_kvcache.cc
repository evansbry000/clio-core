/**
 * workload_llm_kvcache.cc — LLM KV cache offloading for CTE GPU bench
 */

// GPU kernels
#include <cstdint>
#include <cmath>

__global__ void llm_attn_hbm(const float *kv, const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=kv+(uint64_t)h*kvs;
    const float*V=kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd;
    float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for (uint32_t s=0;s<sl;s++){
      float d=0; for(uint32_t i=lid;i<hd;i+=32) d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1) d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0); d/=sqrtf((float)hd);
      if(d>mx){mx=d;bp=s;}
    }
    for(uint32_t i=lid;i<hd;i+=32) O[i]=V[bp*hd+i];
    __syncwarp();
  }
}

__global__ void llm_attn_direct(const float *h_kv, const float *q, float *out,
                                 uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=h_kv+(uint64_t)h*kvs;
    const float*V=h_kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd;
    float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){
      float d=0; for(uint32_t i=lid;i<hd;i+=32) d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1) d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0); d/=sqrtf((float)hd);
      if(d>mx){mx=d;bp=s;}
    }
    for(uint32_t i=lid;i<hd;i+=32) O[i]=V[bp*hd+i];
    __syncwarp();
  }
}

__global__ void llm_kv_wb_hbm(float *kv, const float *nk, const float *nv,
                               uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    for(uint32_t i=lid;i<hd;i+=32) {
      kv[h*kvs+pos*hd+i]=nk[h*hd+i];
      kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];
    }
    __syncwarp();
  }
}

__global__ void llm_kv_wb_direct(float *h_kv, const float *nk, const float *nv,
                                  uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  uint32_t lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    for(uint32_t i=lid;i<hd;i+=32) {
      h_kv[h*kvs+pos*hd+i]=nk[h*hd+i];
      h_kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];
    }
    __syncwarp();
  }
}

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <vector>
#include <cstring>

int run_workload_llm_kvcache(const WorkloadConfig &cfg, const char *mode,
                             WorkloadResult *result) {
  uint32_t nl=cfg.param_num_layers, nh=cfg.param_num_heads;
  uint32_t hd=cfg.param_head_dim, sl=cfg.param_seq_len;
  uint32_t dt=cfg.param_decode_tokens;
  std::string m(mode);

  uint64_t kvpl=2ULL*nh*sl*hd;
  uint64_t kvb=(uint64_t)nl*kvpl*sizeof(float);
  uint64_t qof=(uint64_t)nh*hd;

  float *d_q,*d_o,*d_nk,*d_nv;
  cudaMalloc(&d_q,qof*4); cudaMalloc(&d_o,qof*4);
  cudaMalloc(&d_nk,qof*4); cudaMalloc(&d_nv,qof*4);
  std::vector<float> hq(qof,0.1f);
  cudaMemcpy(d_q,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nk,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv,hq.data(),qof*4,cudaMemcpyHostToDevice);

  uint32_t threads=256;
  uint32_t blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks) blocks=1;

  if (m=="hbm" || m=="cte") {
    float *d_kv; cudaMalloc(&d_kv,kvb); cudaMemset(d_kv,0,kvb);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){
      for(uint32_t l=0;l<nl;l++){
        float*lkv=d_kv+(uint64_t)l*kvpl;
        llm_attn_hbm<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
        llm_kv_wb_hbm<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
      }
    }
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);
    result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvpl*4+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFree(d_kv);
  } else if (m=="direct") {
    float *h_kv; cudaMallocHost(&h_kv,kvb); memset(h_kv,0,kvb);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){
      for(uint32_t l=0;l<nl;l++){
        float*lkv=h_kv+(uint64_t)l*kvpl;
        llm_attn_direct<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
        llm_kv_wb_direct<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
      }
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);
    result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvpl*4+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFreeHost(h_kv);
  } else {
    HLOG(kError,"llm_kvcache: unknown mode '{}'",mode);
    cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
    return -1;
  }
  cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
  return 0;
}

#endif  // HSHM_IS_HOST
