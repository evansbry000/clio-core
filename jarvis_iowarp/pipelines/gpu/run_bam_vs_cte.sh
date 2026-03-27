#!/bin/bash
#
# BaM vs CTE GPU Benchmark — Cache Size Comparison
#
# Compares BaM page cache against CTE (Local + ToLocalCpu) with:
#   - HBM cache ratios: 100%, 50%, 20% (unified --hbm-cache parameter)
#   - Warp I/O sizes: 16KB, 128KB, 16MB
#   - Fixed: client_blocks=64, rt_blocks=16
#
# Output: TSV to stdout, logs to stderr
#
# Usage:
#   bash run_bam_vs_cte.sh [--output-dir DIR]

set -euo pipefail

OUTPUT_DIR="${HOME}/bam_vs_cte_results"
BENCH_BIN="wrp_cte_gpu_bench"
REPEAT=3
TIMEOUT=120
CLIENT_BLOCKS=64
RT_BLOCKS=16

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --bin)        BENCH_BIN="$2"; shift 2 ;;
    --repeat)     REPEAT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="${OUTPUT_DIR}/results.tsv"

# Print TSV header
echo -e "mode\trouting\tio_size\thbm_cache\trun\telapsed_ms\tbandwidth_gbps\tmetric_name\tmetric_value" > "$RESULTS_FILE"

# I/O sizes and corresponding iteration counts
IO_SIZES=("16k" "128k" "16m")
ITERATIONS=(256 64 64)

# HBM cache ratios (unified parameter for both CTE and BaM)
HBM_CACHE_PCTS=(100 50 20)

# Modes to test
MODES=("cte" "cte" "bam" "hbm" "direct")
ROUTINGS=("local" "to_cpu" "local" "local" "local")
MODE_LABELS=("cte_local" "cte_to_cpu" "bam" "hbm" "direct")

run_count=0
total_runs=$(( ${#MODES[@]} * ${#IO_SIZES[@]} * ${#HBM_CACHE_PCTS[@]} * REPEAT ))
# hbm/direct only run once (100%), subtract the extra cache ratios
baseline_modes=2  # hbm + direct
extra_cache_runs=$(( baseline_modes * ${#IO_SIZES[@]} * (${#HBM_CACHE_PCTS[@]} - 1) * REPEAT ))
total_runs=$(( total_runs - extra_cache_runs ))

echo "Total benchmark configurations: $total_runs" >&2
echo "Fixed: client_blocks=$CLIENT_BLOCKS, rt_blocks=$RT_BLOCKS" >&2

for mode_idx in "${!MODES[@]}"; do
  mode="${MODES[$mode_idx]}"
  routing="${ROUTINGS[$mode_idx]}"
  mode_label="${MODE_LABELS[$mode_idx]}"

  for io_idx in "${!IO_SIZES[@]}"; do
    io_size="${IO_SIZES[$io_idx]}"
    iters="${ITERATIONS[$io_idx]}"

    # For hbm and direct modes, cache ratio doesn't apply — run once at 100%
    if [[ "$mode" == "hbm" || "$mode" == "direct" ]]; then
      cache_list=(100)
    else
      cache_list=("${HBM_CACHE_PCTS[@]}")
    fi

    for hbm_cache in "${cache_list[@]}"; do
      for run in $(seq 1 $REPEAT); do
        run_count=$((run_count + 1))
        echo "[$run_count/$total_runs] ${mode_label} io=${io_size} hbm_cache=${hbm_cache}% run=${run}" >&2

        args=(
          --test-case synthetic
          --workload-mode "$mode"
          --routing "$routing"
          --io-size "$io_size"
          --iterations "$iters"
          --rt-blocks "$RT_BLOCKS"
          --rt-threads 32
          --client-blocks "$CLIENT_BLOCKS"
          --client-threads 32
          --hbm-cache "$hbm_cache"
          --timeout "$TIMEOUT"
        )

        # Run benchmark and capture output (strip ANSI codes, filter results)
        raw_output=$("$BENCH_BIN" "${args[@]}" 2>&1 || echo "FAILED")
        output=$(echo "$raw_output" | sed 's/\x1b\[[0-9;]*m//g' | grep -E '(Elapsed:|Bandwidth:|=== |putgets|edges|tokens|ms/step|nodes)' || echo "")

        if echo "$raw_output" | grep -q "FAILED"; then
          echo -e "${mode_label}\t${routing}\t${io_size}\t${hbm_cache}\t${run}\t-1\t-1\t\t" >> "$RESULTS_FILE"
          echo "  FAILED" >&2
        else
          # Parse results from stdout
          elapsed=$(echo "$output" | grep -oP 'Elapsed:\s+\K[\d.]+' || echo "-1")
          bw=$(echo "$output" | grep -oP 'Bandwidth:\s+\K[\d.]+' || echo "-1")
          metric_line=$(echo "$output" | grep -E '^\S+\s+[\d.e+-]+$' || echo "")
          metric_name=$(echo "$metric_line" | awk '{print $1}' || echo "")
          metric_val=$(echo "$metric_line" | awk '{print $2}' || echo "")

          echo -e "${mode_label}\t${routing}\t${io_size}\t${hbm_cache}\t${run}\t${elapsed}\t${bw}\t${metric_name}\t${metric_val}" >> "$RESULTS_FILE"
          echo "  elapsed=${elapsed}ms bw=${bw}GB/s" >&2
        fi
      done
    done
  done
done

echo "" >&2
echo "Results written to: $RESULTS_FILE" >&2
echo "Total runs: $run_count" >&2
