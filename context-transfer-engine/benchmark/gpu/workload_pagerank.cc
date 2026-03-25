/**
 * workload_pagerank.cc — PageRank graph algorithm for CTE GPU bench
 */

// GPU kernels (visible in both host and device compilation passes)
#include <cstdint>
#include <cmath>

__global__ void pr_push_hbm(const uint64_t *offsets, const uint32_t *edges,
                             const float *values, float *residuals,
                             uint32_t nv, float alpha) {
  uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lid = threadIdx.x & 31;
  uint32_t nw = (blockDim.x * gridDim.x) / 32;
  for (uint32_t v = wid; v < nv; v += nw) {
    uint64_t s = offsets[v], e = offsets[v + 1];
    uint32_t deg = (uint32_t)(e - s);
    if (deg == 0) continue;
    float contrib = alpha * values[v] / deg;
    for (uint64_t i = s + lid; i < e; i += 32)
      atomicAdd(&residuals[edges[i]], contrib);
    __syncwarp();
  }
}

__global__ void pr_push_direct(const uint64_t *offsets, const uint32_t *h_edges,
                                const float *values, float *residuals,
                                uint32_t nv, float alpha) {
  uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lid = threadIdx.x & 31;
  uint32_t nw = (blockDim.x * gridDim.x) / 32;
  for (uint32_t v = wid; v < nv; v += nw) {
    uint64_t s = offsets[v], e = offsets[v + 1];
    uint32_t deg = (uint32_t)(e - s);
    if (deg == 0) continue;
    float contrib = alpha * values[v] / deg;
    for (uint64_t i = s + lid; i < e; i += 32)
      atomicAdd(&residuals[h_edges[i]], contrib);
    __syncwarp();
  }
}

__global__ void pr_update(float *values, float *residuals,
                           uint32_t nv, float tol, int *active) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nv) return;
  float r = residuals[tid];
  if (fabsf(r) > tol) {
    values[tid] += r;
    atomicAdd(active, 1);
  }
  residuals[tid] = 0.0f;
}

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

struct CSRGraph {
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> edges;
  uint32_t num_vertices;
  uint64_t num_edges;
};

static CSRGraph gen_rmat(uint32_t nv, uint32_t avg_deg, uint64_t seed = 42) {
  CSRGraph g;
  g.num_vertices = nv;
  std::mt19937_64 rng(seed);
  std::vector<std::vector<uint32_t>> adj(nv);
  uint64_t target = (uint64_t)nv * avg_deg;
  double a = 0.57, b = 0.19, c = 0.19;
  uint32_t log2n = 0;
  for (uint32_t n = nv; n > 1; n >>= 1) log2n++;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint64_t e = 0; e < target; e++) {
    uint32_t u = 0, v = 0;
    for (uint32_t l = 0; l < log2n; l++) {
      double r = dist(rng);
      uint32_t half = 1u << (log2n - l - 1);
      if (r < a) {} else if (r < a + b) v += half;
      else if (r < a + b + c) u += half; else { u += half; v += half; }
    }
    u %= nv; v %= nv;
    if (u != v) adj[u].push_back(v);
  }
  g.offsets.resize(nv + 1); g.offsets[0] = 0;
  for (uint32_t i = 0; i < nv; i++) {
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    g.offsets[i + 1] = g.offsets[i] + adj[i].size();
  }
  g.num_edges = g.offsets[nv];
  g.edges.resize(g.num_edges);
  for (uint32_t i = 0; i < nv; i++)
    std::copy(adj[i].begin(), adj[i].end(), g.edges.begin() + g.offsets[i]);
  return g;
}

int run_workload_pagerank(const WorkloadConfig &cfg, const char *mode,
                          WorkloadResult *result) {
  uint32_t nv = cfg.param_vertices;
  int iters = cfg.iterations > 0 ? cfg.iterations : 10;
  float alpha = 0.85f, tol = 0.001f;
  std::string m(mode);

  HIPRINT("  Generating R-MAT graph: {} vertices, avg degree {}", nv, cfg.param_avg_degree);
  CSRGraph g = gen_rmat(nv, cfg.param_avg_degree);
  HIPRINT("  Graph: {} edges ({:.1f} MB)", g.num_edges, g.num_edges * 4 / (1024.0 * 1024.0));

  uint64_t *d_off;
  cudaMalloc(&d_off, (nv + 1) * sizeof(uint64_t));
  cudaMemcpy(d_off, g.offsets.data(), (nv + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

  float *d_vals, *d_res;
  cudaMalloc(&d_vals, nv * sizeof(float));
  cudaMalloc(&d_res, nv * sizeof(float));
  std::vector<float> ones(nv, 1.0f);
  cudaMemcpy(d_vals, ones.data(), nv * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_res, 0, nv * sizeof(float));

  int *d_active;
  cudaMallocHost(&d_active, sizeof(int));

  uint32_t threads = 256;
  uint32_t blocks = (cfg.client_blocks * cfg.client_threads + threads - 1) / threads;
  if (blocks == 0) blocks = 1;
  int ublocks = (nv + 255) / 256;

  if (m == "hbm" || m == "cte") {
    uint32_t *d_edges;
    cudaMalloc(&d_edges, g.num_edges * sizeof(uint32_t));
    cudaMemcpy(d_edges, g.edges.data(), g.num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto t0 = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < iters; iter++) {
      pr_push_hbm<<<blocks, threads>>>(d_off, d_edges, d_vals, d_res, nv, alpha);
      cudaDeviceSynchronize();
      *d_active = 0;
      pr_update<<<ublocks, 256>>>(d_vals, d_res, nv, tol, d_active);
      cudaDeviceSynchronize();
      if (*d_active == 0) { iter++; break; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result->primary_metric = (double)g.num_edges * iter / (result->elapsed_ms / 1e3);
    result->metric_name = "edges/sec";
    result->bandwidth_gbps = (g.num_edges * 4.0 * iter / 1e9) / (result->elapsed_ms / 1e3);
    cudaFree(d_edges);
  } else if (m == "direct") {
    uint32_t *h_edges;
    cudaMallocHost(&h_edges, g.num_edges * sizeof(uint32_t));
    memcpy(h_edges, g.edges.data(), g.num_edges * sizeof(uint32_t));
    auto t0 = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < iters; iter++) {
      pr_push_direct<<<blocks, threads>>>(d_off, h_edges, d_vals, d_res, nv, alpha);
      cudaDeviceSynchronize();
      *d_active = 0;
      pr_update<<<ublocks, 256>>>(d_vals, d_res, nv, tol, d_active);
      cudaDeviceSynchronize();
      if (*d_active == 0) { iter++; break; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result->primary_metric = (double)g.num_edges * iter / (result->elapsed_ms / 1e3);
    result->metric_name = "edges/sec";
    result->bandwidth_gbps = (g.num_edges * 4.0 * iter / 1e9) / (result->elapsed_ms / 1e3);
    cudaFreeHost(h_edges);
  } else {
    HLOG(kError, "pagerank: unknown mode '{}'", mode);
    cudaFree(d_off); cudaFree(d_vals); cudaFree(d_res); cudaFreeHost(d_active);
    return -1;
  }

  cudaFree(d_off); cudaFree(d_vals); cudaFree(d_res); cudaFreeHost(d_active);
  return 0;
}

#endif  // HSHM_IS_HOST
