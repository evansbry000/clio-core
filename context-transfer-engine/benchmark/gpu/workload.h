/**
 * workload.h — Interface for GPU workload benchmarks
 *
 * Each workload (pagerank, gnn, gray_scott, llm_kvcache) provides
 * a run function for each mode. These are called from the main
 * wrp_cte_gpu_bench driver.
 *
 * CTE mode requires the Chimaera runtime to be initialized.
 * BaM and direct modes are standalone.
 */
#ifndef BENCH_GPU_WORKLOAD_H
#define BENCH_GPU_WORKLOAD_H

#if HSHM_IS_HOST

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/bam.h>
#endif

#include <cstdint>
#include <string>

struct WorkloadConfig {
  // Grid/thread config
  uint32_t rt_blocks = 1;
  uint32_t rt_threads = 32;
  uint32_t client_blocks = 4;
  uint32_t client_threads = 256;
  uint32_t iterations = 1;
  int timeout_sec = 60;

  // BaM config
  uint64_t bam_page_size = 65536;
  uint32_t bam_cache_pages = 512;

  // CTE config (set by main driver after Chimaera init)
  chi::PoolId cte_pool_id;
  wrp_cte::core::TagId tag_id;
  std::string bdev_type = "pinned";

  // Workload-specific parameters (encoded as key=value strings)
  // Parsed by each workload's run function
  uint32_t param_vertices = 100000;
  uint32_t param_avg_degree = 16;
  uint32_t param_num_nodes = 500000;
  uint32_t param_emb_dim = 128;
  uint32_t param_batch_size = 1024;
  uint32_t param_grid_size = 128;
  uint32_t param_steps = 100;
  uint32_t param_checkpoint_freq = 10;
  uint32_t param_num_layers = 12;
  uint32_t param_num_heads = 12;
  uint32_t param_head_dim = 64;
  uint32_t param_seq_len = 2048;
  uint32_t param_decode_tokens = 32;
};

struct WorkloadResult {
  double elapsed_ms;
  double bandwidth_gbps;
  double primary_metric;
  const char *metric_name;
};

// Workload run functions — each returns 0 on success
// mode: "cte", "bam", "direct", "hbm"

int run_workload_pagerank(const WorkloadConfig &cfg, const char *mode,
                          WorkloadResult *result);

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode,
                     WorkloadResult *result);

int run_workload_gray_scott(const WorkloadConfig &cfg, const char *mode,
                            WorkloadResult *result);

int run_workload_llm_kvcache(const WorkloadConfig &cfg, const char *mode,
                             WorkloadResult *result);

#endif  // HSHM_IS_HOST
#endif  // BENCH_GPU_WORKLOAD_H
