#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <cmath>
#include <fstream>
#include <chrono>

// clio-core and hardware headers
#include "chimaera/chimaera.h"
#include "pi_timer.h" 

using namespace chi;

// Trace event structure
struct TraceEvent {
    uint64_t start_tick;
    uint64_t end_tick;
};

// Pareto distribution generator
double pareto_random(std::mt19937& gen, double scale, double shape) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(gen);
    if (u == 0.0) u = 0.0001;
    return scale / std::pow(u, 1.0 / shape);
}

// Global control
std::atomic<bool> keep_running{true};

// Noisy Neighbor: Async heavy-tail workload
void NoisyNeighborThread() {
    std::cout << "[NoisyNeighbor] Started" << std::endl;
    std::mt19937 gen(42);
    
    size_t count = 0;
    while (keep_running.load(std::memory_order_relaxed)) {
        double compute_cost = pareto_random(gen, 1000.0, 1.5);
        if (count++ % 1000 == 0) {
            std::cout << "[NoisyNeighbor] Spike: " << compute_cost << " ms" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::cout << "[NoisyNeighbor] Completed " << count << " iterations" << std::endl;
}

// Micro-I/O Barrage: Sync measured latency
void MicroIoBarrageThread(std::vector<TraceEvent>& trace_buffer, size_t num_events) {
    std::cout << "[MicroIoBarrage] Started" << std::endl;
    std::mt19937 gen(1337);
    std::exponential_distribution<double> poisson_arrival(1.0 / 200.0);

    for (size_t i = 0; i < num_events; ++i) {
        double delay_us = poisson_arrival(gen);
        
        auto start_wait = get_cntvct_el0();
        uint64_t wait_ticks = static_cast<uint64_t>(delay_us * 54.0);
        while ((get_cntvct_el0() - start_wait) < wait_ticks) {
            // Busy-wait
        }

        trace_buffer[i].start_tick = get_cntvct_el0();
        
        // Simulate 2us task processing
        std::this_thread::sleep_for(std::chrono::microseconds(2));
        
        trace_buffer[i].end_tick = get_cntvct_el0();
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "[MicroIoBarrage] " << (i + 1) << " / " << num_events << std::endl;
        }
    }
    
    keep_running.store(false, std::memory_order_relaxed);
    std::cout << "[MicroIoBarrage] Completed" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "[Benchmark] Starting Edge Benchmark" << std::endl;

    uint64_t calib = chi::calibrate_timer_overhead();
    std::cout << "[Benchmark] Timer overhead: " << calib << " ticks" << std::endl;

    size_t test_size = 100000;
    std::vector<TraceEvent> trace_buffer(test_size);

    std::thread noisy_neighbor(NoisyNeighborThread);
    std::thread micro_io(MicroIoBarrageThread, std::ref(trace_buffer), test_size);

    micro_io.join();
    noisy_neighbor.join();

    std::cout << "[Benchmark] Writing trace_results.csv" << std::endl;
    std::ofstream outfile("trace_results.csv");
    outfile << "task_id,latency_ticks,latency_us\n";
    
    for (size_t i = 0; i < test_size; ++i) {
        uint64_t elapsed_ticks = trace_buffer[i].end_tick - trace_buffer[i].start_tick;
        double elapsed_us = static_cast<double>(elapsed_ticks) / 54.0;
        outfile << i << "," << elapsed_ticks << "," << elapsed_us << "\n";
    }
    outfile.close();

    std::cout << "[Benchmark] Complete" << std::endl;
    return 0;
}