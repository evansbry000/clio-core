/**
 * workload_gnn.cc — GNN feature loading for CTE GPU bench
 */

// GPU kernels
#include <cstdint>

__global__ void gnn_gather_hbm(const float *features,
                                const uint32_t *indices, float *output,
                                uint32_t batch_size, uint32_t emb_dim) {
  uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lid = threadIdx.x & 31;
  uint32_t nw = (blockDim.x * gridDim.x) / 32;
  for (uint32_t b = wid; b < batch_size; b += nw) {
    uint32_t node = indices[b];
    const float *in = features + (uint64_t)node * emb_dim;
    float *out = output + (uint64_t)b * emb_dim;
    for (uint32_t f = lid; f < emb_dim; f += 32) out[f] = in[f];
    __syncwarp();
  }
}

__global__ void gnn_gather_direct(const float *h_features,
                                   const uint32_t *indices, float *output,
                                   uint32_t batch_size, uint32_t emb_dim) {
  uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lid = threadIdx.x & 31;
  uint32_t nw = (blockDim.x * gridDim.x) / 32;
  for (uint32_t b = wid; b < batch_size; b += nw) {
    uint32_t node = indices[b];
    const float *in = h_features + (uint64_t)node * emb_dim;
    float *out = output + (uint64_t)b * emb_dim;
    for (uint32_t f = lid; f < emb_dim; f += 32) out[f] = in[f];
    __syncwarp();
  }
}

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <vector>
#include <random>
#include <cstring>

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode,
                     WorkloadResult *result) {
  uint32_t nn = cfg.param_num_nodes, ed = cfg.param_emb_dim;
  uint32_t bs = cfg.param_batch_size;
  uint32_t nb = cfg.iterations > 0 ? cfg.iterations : 10;
  std::string m(mode);
  uint64_t feat_bytes = (uint64_t)nn * ed * sizeof(float);

  uint32_t threads = 256;
  uint32_t blocks = (cfg.client_blocks * cfg.client_threads + threads - 1) / threads;
  if (blocks == 0) blocks = 1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> ndist(0, nn - 1);

  uint32_t *d_idx; float *d_out;
  cudaMalloc(&d_idx, bs * sizeof(uint32_t));
  cudaMalloc(&d_out, (uint64_t)bs * ed * sizeof(float));
  std::vector<uint32_t> h_idx(bs);

  if (m == "hbm" || m == "cte") {
    float *d_feat;
    cudaMalloc(&d_feat, feat_bytes);
    std::vector<float> h_feat((uint64_t)nn * ed);
    for (auto &x : h_feat) x = fdist(rng);
    cudaMemcpy(d_feat, h_feat.data(), feat_bytes, cudaMemcpyHostToDevice);

    // Warmup
    for (uint32_t i = 0; i < bs; i++) h_idx[i] = ndist(rng);
    cudaMemcpy(d_idx, h_idx.data(), bs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    gnn_gather_hbm<<<blocks, threads>>>(d_feat, d_idx, d_out, bs, ed);
    cudaDeviceSynchronize();

    double total_ms = 0;
    for (uint32_t b = 0; b < nb; b++) {
      for (uint32_t i = 0; i < bs; i++) h_idx[i] = ndist(rng);
      cudaMemcpy(d_idx, h_idx.data(), bs * sizeof(uint32_t), cudaMemcpyHostToDevice);
      auto t0 = std::chrono::high_resolution_clock::now();
      gnn_gather_hbm<<<blocks, threads>>>(d_feat, d_idx, d_out, bs, ed);
      cudaDeviceSynchronize();
      auto t1 = std::chrono::high_resolution_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    result->elapsed_ms = total_ms / nb;
    result->bandwidth_gbps = ((uint64_t)bs * ed * 4 / 1e9) / (result->elapsed_ms / 1e3);
    result->primary_metric = bs / (result->elapsed_ms / 1e3);
    result->metric_name = "nodes/sec";
    cudaFree(d_feat);

  } else if (m == "direct") {
    float *h_feat;
    cudaMallocHost(&h_feat, feat_bytes);
    for (uint64_t i = 0; i < (uint64_t)nn * ed; i++) h_feat[i] = fdist(rng);

    for (uint32_t i = 0; i < bs; i++) h_idx[i] = ndist(rng);
    cudaMemcpy(d_idx, h_idx.data(), bs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    gnn_gather_direct<<<blocks, threads>>>(h_feat, d_idx, d_out, bs, ed);
    cudaDeviceSynchronize();

    double total_ms = 0;
    for (uint32_t b = 0; b < nb; b++) {
      for (uint32_t i = 0; i < bs; i++) h_idx[i] = ndist(rng);
      cudaMemcpy(d_idx, h_idx.data(), bs * sizeof(uint32_t), cudaMemcpyHostToDevice);
      auto t0 = std::chrono::high_resolution_clock::now();
      gnn_gather_direct<<<blocks, threads>>>(h_feat, d_idx, d_out, bs, ed);
      cudaDeviceSynchronize();
      auto t1 = std::chrono::high_resolution_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    result->elapsed_ms = total_ms / nb;
    result->bandwidth_gbps = ((uint64_t)bs * ed * 4 / 1e9) / (result->elapsed_ms / 1e3);
    result->primary_metric = bs / (result->elapsed_ms / 1e3);
    result->metric_name = "nodes/sec";
    cudaFreeHost(h_feat);
  } else {
    HLOG(kError, "gnn: unknown mode '{}'", mode);
    cudaFree(d_idx); cudaFree(d_out);
    return -1;
  }
  cudaFree(d_idx); cudaFree(d_out);
  return 0;
}

#endif  // HSHM_IS_HOST
