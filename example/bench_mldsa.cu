#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <functional>
#include <map>

using namespace cupqc;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ML-DSA type definitions
using MLDSA44Key = decltype(ML_DSA_44()
                            + Function<function::Keygen>()
                            + Block()
                            + BlockDim<128>());

using MLDSA44Sign = decltype(ML_DSA_44()
                             + Function<function::Sign>()
                             + Block()
                             + BlockDim<128>());

using MLDSA44Verify = decltype(ML_DSA_44()
                               + Function<function::Verify>()
                               + Block()
                               + BlockDim<128>());

// Your existing kernels
__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* randombytes, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLDSA44Key::public_key_size;
    auto secret_key = secret_keys + block * MLDSA44Key::secret_key_size;
    auto entropy    = randombytes + block * MLDSA44Key::entropy_size;
    auto work       = workspace   + block * MLDSA44Key::workspace_size;

    MLDSA44Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void sign_kernel(uint8_t* signatures, const uint8_t* messages, const size_t message_size, const uint8_t* secret_keys, uint8_t *randombytes, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Sign::shared_memory_size];
    int block = blockIdx.x;
    auto signature  = signatures  + block * ((MLDSA44Sign::signature_size + 7) / 8 * 8);
    auto message    = messages    + block * message_size;
    auto secret_key = secret_keys + block * MLDSA44Sign::secret_key_size;
    auto entropy    = randombytes + block * MLDSA44Sign::entropy_size;
    auto work       = workspace   + block * MLDSA44Sign::workspace_size;

    MLDSA44Sign().execute(signature, message, message_size, secret_key, entropy, work, smem_ptr);
}

__global__ void verify_kernel(bool* valids, const uint8_t* signatures, const uint8_t* messages, const size_t message_size, const uint8_t* public_keys, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Verify::shared_memory_size];
    int block = blockIdx.x;
    auto signature   = signatures  + block * ((MLDSA44Sign::signature_size + 7) / 8 * 8);
    auto message     = messages    + block * message_size;
    auto public_key  = public_keys + block * MLDSA44Verify::public_key_size;
    auto work        = workspace   + block * MLDSA44Verify::workspace_size;

    valids[block] = MLDSA44Verify().execute(message, message_size, signature, public_key, work, smem_ptr);
}

struct GPUInfo {
    std::string name;
    int sm_count;
    size_t total_memory;
    size_t free_memory;
    double memory_bandwidth_gbps;
};

struct BenchmarkResult {
    std::string operation_name;
    unsigned int batch_size;
    double avg_time_ms;
    double throughput_ops_per_sec;
    double memory_bandwidth_gbps;
    double gpu_utilization_percent;
    size_t memory_usage_gb;
};

class MaxPerformanceBenchmark {
private:
    GPUInfo gpu_info;
    std::vector<BenchmarkResult> results;

public:
    MaxPerformanceBenchmark() {
        initializeGPU();
        printf("🚀 ML-DSA GH200 Maximum Performance Benchmarking\n");
        printf("Device: %s | SMs: %d | Memory: %.1f GB\n\n", 
               gpu_info.name.c_str(), gpu_info.sm_count, gpu_info.total_memory / 1e9);
    }

    void initializeGPU() {
        CUDA_CHECK(cudaSetDevice(0));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        gpu_info.name = std::string(prop.name);
        gpu_info.sm_count = prop.multiProcessorCount;
        gpu_info.total_memory = total_mem;
        gpu_info.free_memory = free_mem;
        gpu_info.memory_bandwidth_gbps = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    }

    std::vector<unsigned int> generateOptimalBatchSizes(size_t message_size) {
        // Calculate memory for single ML-DSA operation
        size_t signature_padded = ((MLDSA44Sign::signature_size + 7) / 8) * 8;
        size_t single_op_memory = MLDSA44Key::public_key_size + MLDSA44Key::secret_key_size + 
                                 MLDSA44Key::workspace_size + MLDSA44Key::entropy_size +
                                 signature_padded + message_size;
        
        size_t usable_memory = static_cast<size_t>(gpu_info.free_memory * 0.75); // Conservative 75%
        unsigned int max_memory_batch = usable_memory / single_op_memory;
        
        // Cap maximum batch size to something reasonable
        unsigned int REASONABLE_MAX = 500000; // 500K operations max
        max_memory_batch = std::min(max_memory_batch, REASONABLE_MAX);
        
        unsigned int sm_4x = gpu_info.sm_count * 4;
        unsigned int sm_8x = gpu_info.sm_count * 8;
        unsigned int sm_16x = gpu_info.sm_count * 16;
        
        std::vector<unsigned int> batch_sizes = {
            10000, 25000, 50000, 100000, 200000, // More reasonable progression
            sm_4x, sm_8x, sm_16x,
            max_memory_batch / 4, max_memory_batch / 2,
            static_cast<unsigned int>(max_memory_batch * 0.75),
            max_memory_batch  // Now capped at 500K max
        };
        
        std::sort(batch_sizes.begin(), batch_sizes.end());
        batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()), batch_sizes.end());
        
        std::vector<unsigned int> valid_batches;
        for (auto batch : batch_sizes) {
            if (batch > 0 && batch * single_op_memory <= usable_memory && batch <= REASONABLE_MAX) {
                valid_batches.push_back(batch);
            }
        }
        
        printf("📊 Generated %zu safe batch configurations (max: %u ops, %.1f GB memory)\n", 
               valid_batches.size(), REASONABLE_MAX, (REASONABLE_MAX * single_op_memory) / 1e9);
        
        return valid_batches;
    }

    BenchmarkResult runSilentBenchmark(const std::string& operation_name,
                                      std::function<void()> kernel_func,
                                      unsigned int batch_size,
                                      int iterations,
                                      size_t memory_bytes_per_op) {
        
        std::vector<float> iteration_times;
        iteration_times.reserve(iterations);
        
        // Silent warm-up
        for (int i = 0; i < 10; i++) {
            kernel_func();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Silent benchmark timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        for (int i = 0; i < iterations; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            kernel_func();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
            iteration_times.push_back(time_ms);
        }
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Calculate statistics
        std::sort(iteration_times.begin(), iteration_times.end());
        double sum = 0.0;
        for (float time : iteration_times) sum += time;
        double avg_time = sum / iteration_times.size();
        
        double avg_time_seconds = avg_time / 1000.0;
        double throughput = batch_size / avg_time_seconds;
        double bytes_per_iteration = memory_bytes_per_op * batch_size;
        double memory_bandwidth = (bytes_per_iteration / avg_time_seconds) / 1e9;
        double gpu_utilization = (memory_bandwidth / gpu_info.memory_bandwidth_gbps) * 100.0;
        
        BenchmarkResult result = {
            operation_name,
            batch_size,
            avg_time,
            throughput,
            memory_bandwidth,
            gpu_utilization,
            static_cast<size_t>(bytes_per_iteration / 1e9)
        };
        
        results.push_back(result);
        return result;
    }

    const std::vector<BenchmarkResult>& getResults() const {
        return results;
    }
};

// Safe ML-DSA functions with better error handling
bool ml_dsa_keygen_safe(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, 
                       unsigned int batch, int iterations, MaxPerformanceBenchmark& benchmark) {
    
    auto length_public_key = MLDSA44Key::public_key_size;
    auto length_secret_key = MLDSA44Key::secret_key_size;

    auto workspace = make_workspace<MLDSA44Key>(batch);
    auto randombytes = get_entropy<MLDSA44Key>(batch);

    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    // Safe allocation with error handling
    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation) {
        if (d_public_key) cudaFree(d_public_key);
        if (d_secret_key) cudaFree(d_secret_key);
        destroy_workspace(workspace);
        release_entropy(randombytes);
        return false; // Out of memory
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);

    size_t memory_ops = (length_public_key + length_secret_key + 
                        MLDSA44Key::workspace_size + MLDSA44Key::entropy_size);

    auto kernel_func = [&]() {
        keygen_kernel<<<batch, MLDSA44Key::BlockDim>>>(d_public_key, d_secret_key, randombytes, workspace);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Key Generation", kernel_func, batch, iterations, memory_ops);

    CUDA_CHECK(cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_public_key));
    CUDA_CHECK(cudaFree(d_secret_key));
    destroy_workspace(workspace);
    release_entropy(randombytes);
    CUDA_CHECK(cudaDeviceSynchronize()); // Force cleanup
    
    return true;
}

bool ml_dsa_sign_safe(std::vector<uint8_t>& signatures, std::vector<uint8_t>& messages, size_t message_size,
                     const std::vector<uint8_t>& secret_keys, unsigned int batch, int iterations, 
                     MaxPerformanceBenchmark& benchmark) {
    
    auto length_secret_key = MLDSA44Sign::secret_key_size;
    auto length_signature = MLDSA44Sign::signature_size;
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;

    auto workspace = make_workspace<MLDSA44Sign>(batch);
    auto randombytes = get_entropy<MLDSA44Sign>(batch);

    uint8_t* d_signatures = nullptr;
    uint8_t* d_secret_keys = nullptr;
    uint8_t* d_messages = nullptr;

    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_signatures), length_signature_padded * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_secret_keys), length_secret_key * batch);
    cudaError_t error3 = cudaMalloc(reinterpret_cast<void**>(&d_messages), message_size * batch);
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation || error3 == cudaErrorMemoryAllocation) {
        if (d_signatures) cudaFree(d_signatures);
        if (d_secret_keys) cudaFree(d_secret_keys);
        if (d_messages) cudaFree(d_messages);
        destroy_workspace(workspace);
        release_entropy(randombytes);
        return false;
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);
    CUDA_CHECK(error3);

    CUDA_CHECK(cudaMemcpy(d_secret_keys, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice));

    size_t memory_ops = (length_signature_padded + length_secret_key + message_size +
                        MLDSA44Sign::workspace_size + MLDSA44Sign::entropy_size);

    auto kernel_func = [&]() {
        sign_kernel<<<batch, MLDSA44Sign::BlockDim>>>(d_signatures, d_messages, message_size, d_secret_keys, randombytes, workspace);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Signing", kernel_func, batch, iterations, memory_ops);

    CUDA_CHECK(cudaMemcpy(signatures.data(), d_signatures, length_signature_padded * batch, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_signatures));
    CUDA_CHECK(cudaFree(d_secret_keys));
    CUDA_CHECK(cudaFree(d_messages));
    destroy_workspace(workspace);
    release_entropy(randombytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return true;
}

bool ml_dsa_verify_safe(std::vector<uint8_t>& is_valids, const std::vector<uint8_t>& signatures, 
                       const std::vector<uint8_t>& messages, size_t message_size,
                       const std::vector<uint8_t>& public_keys, unsigned int batch, int iterations,
                       MaxPerformanceBenchmark& benchmark) {
    
    auto workspace = make_workspace<MLDSA44Verify>(batch);
    auto length_signature = MLDSA44Verify::signature_size;
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;
    auto length_public_key = MLDSA44Verify::public_key_size;

    uint8_t* d_signatures = nullptr;
    uint8_t* d_messages = nullptr;
    uint8_t* d_public_keys = nullptr;
    bool* d_valids = nullptr;

    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_signatures), length_signature_padded * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_public_keys), length_public_key * batch);
    cudaError_t error3 = cudaMalloc(reinterpret_cast<void**>(&d_messages), message_size * batch);
    cudaError_t error4 = cudaMalloc(reinterpret_cast<void**>(&d_valids), batch * sizeof(bool));
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation || 
        error3 == cudaErrorMemoryAllocation || error4 == cudaErrorMemoryAllocation) {
        if (d_signatures) cudaFree(d_signatures);
        if (d_public_keys) cudaFree(d_public_keys);
        if (d_messages) cudaFree(d_messages);
        if (d_valids) cudaFree(d_valids);
        destroy_workspace(workspace);
        return false;
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);
    CUDA_CHECK(error3);
    CUDA_CHECK(error4);

    CUDA_CHECK(cudaMemcpy(d_public_keys, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_signatures, signatures.data(), length_signature_padded * batch, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice));

    size_t memory_ops = (length_signature_padded + length_public_key + message_size + sizeof(bool) +
                        MLDSA44Verify::workspace_size);

    auto kernel_func = [&]() {
        verify_kernel<<<batch, MLDSA44Verify::BlockDim>>>(d_valids, d_signatures, d_messages, message_size, d_public_keys, workspace);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Verification", kernel_func, batch, iterations, memory_ops);

    // **FIXED: Cast to uint8_t* for compatibility with std::vector<uint8_t>**
    CUDA_CHECK(cudaMemcpy(is_valids.data(), reinterpret_cast<uint8_t*>(d_valids), batch * sizeof(bool), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_signatures));
    CUDA_CHECK(cudaFree(d_public_keys));
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_valids));
    destroy_workspace(workspace);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return true;
}

void printFinalResults(const std::vector<BenchmarkResult>& results) {
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                    🏆 GH200 ML-DSA MAXIMUM PERFORMANCE RESULTS 🏆               ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-15s ║ %-10s ║ %-12s ║ %-15s ║ %-12s ║ %-10s ║ %-8s ║\n", 
           "Operation", "Batch", "Time (ms)", "Throughput", "Bandwidth", "GPU Util%", "Mem(GB)");
    printf("║                 ║            ║              ║    (ops/sec)    ║    (GB/s)    ║            ║          ║\n");
    printf("╠═════════════════╬════════════╬══════════════╬═════════════════╬══════════════╬════════════╬══════════╣\n");
    
    // Find best results for each operation
    std::map<std::string, BenchmarkResult> best_results;
    for (const auto& result : results) {
        if (best_results.find(result.operation_name) == best_results.end() || 
            result.throughput_ops_per_sec > best_results[result.operation_name].throughput_ops_per_sec) {
            best_results[result.operation_name] = result;
        }
    }
    
    // Print all results, highlighting best ones
    for (const auto& result : results) {
        bool is_best = (best_results[result.operation_name].batch_size == result.batch_size &&
                       best_results[result.operation_name].throughput_ops_per_sec == result.throughput_ops_per_sec);
        
        std::string marker = is_best ? "🔥" : "  ";
        
        printf("║%s%-13s ║ %-10u ║ %-12.4f ║ %-15.0f ║ %-12.2f ║ %-10.1f ║ %-8zu ║\n",
               marker.c_str(),
               result.operation_name.c_str(),
               result.batch_size,
               result.avg_time_ms,
               result.throughput_ops_per_sec,
               result.memory_bandwidth_gbps,
               result.gpu_utilization_percent,
               result.memory_usage_gb);
    }
    
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Summary of peak performance
    printf("\n🎯 PEAK PERFORMANCE SUMMARY:\n");
    for (const auto& pair : best_results) {
        const auto& result = pair.second;
        printf("   %s: %.0f ops/sec (batch: %u, %.1f%% GPU util)\n",
               result.operation_name.c_str(),
               result.throughput_ops_per_sec,
               result.batch_size,
               result.gpu_utilization_percent);
    }
    printf("\n");
}

int main() {
    MaxPerformanceBenchmark benchmark;
    
    constexpr size_t message_size = 1024; // 1KB messages like in your original
    auto batch_sizes = benchmark.generateOptimalBatchSizes(message_size);
    
    int iterations = 100;
    
    printf("Testing %zu batch configurations with %zu byte messages... ", batch_sizes.size(), message_size);
    fflush(stdout);
    
    for (size_t i = 0; i < batch_sizes.size(); i++) {
        auto batch = batch_sizes[i];
        
        // Progress indicator
        printf(".");
        fflush(stdout);
        
        auto signature_padded = ((MLDSA44Sign::signature_size + 7) / 8) * 8;
        
        std::vector<uint8_t> public_keys(MLDSA44Key::public_key_size * batch);
        std::vector<uint8_t> secret_keys(MLDSA44Key::secret_key_size * batch);
        std::vector<uint8_t> signatures(signature_padded * batch);
        std::vector<uint8_t> messages(message_size * batch);
        std::vector<uint8_t> is_valids(batch); // **FIXED: Changed from std::vector<bool> to std::vector<uint8_t>**
        
        // Initialize messages with some data
        for (size_t j = 0; j < messages.size(); j++) {
            messages[j] = static_cast<uint8_t>(j % 256);
        }
        
        // Safe execution with proper error handling
        bool keygen_success = ml_dsa_keygen_safe(public_keys, secret_keys, batch, iterations, benchmark);
        if (!keygen_success) {
            printf("⚠"); // Memory limit reached
            continue;
        }
        
        bool sign_success = ml_dsa_sign_safe(signatures, messages, message_size, secret_keys, batch, iterations, benchmark);
        if (!sign_success) {
            printf("⚠"); 
            continue;
        }
        
        bool verify_success = ml_dsa_verify_safe(is_valids, signatures, messages, message_size, public_keys, batch, iterations, benchmark);
        if (!verify_success) {
            printf("⚠");
            continue;
        }
    }
    
    printf(" Done!\n\n");
    
    // Show only the final results table
    printFinalResults(benchmark.getResults());
    
    printf("🎉 ML-DSA Benchmarking completed! Your GH200 has been pushed to its limits! 💪\n");
    return 0;
}

