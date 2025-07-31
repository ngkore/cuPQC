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

// ML-KEM-1024 type definitions
using MLKEM1024Key = decltype(ML_KEM_1024()
                             + Function<function::Keygen>()
                             + Block()
                             + BlockDim<128>());

using MLKEM1024Encaps = decltype(ML_KEM_1024()
                                + Function<function::Encaps>()
                                + Block()
                                + BlockDim<128>());

using MLKEM1024Decaps = decltype(ML_KEM_1024()
                                + Function<function::Decaps>()
                                + Block()
                                + BlockDim<128>());

// Kernels
__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM1024Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLKEM1024Key::public_key_size;
    auto secret_key = secret_keys + block * MLKEM1024Key::secret_key_size;
    auto entropy    = randombytes + block * MLKEM1024Key::entropy_size;
    auto work       = workspace   + block * MLKEM1024Key::workspace_size;

    MLKEM1024Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void encaps_kernel(uint8_t* ciphertexts, uint8_t* shared_secrets, const uint8_t* public_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM1024Encaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM1024Encaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM1024Encaps::ciphertext_size;
    auto public_key    = public_keys + block * MLKEM1024Encaps::public_key_size;
    auto entropy       = randombytes + block * MLKEM1024Encaps::entropy_size;
    auto work          = workspace   + block * MLKEM1024Encaps::workspace_size;

    MLKEM1024Encaps().execute(ciphertext, shared_secret, public_key, entropy, work, smem_ptr);
}

__global__ void decaps_kernel(uint8_t* shared_secrets, const uint8_t* ciphertexts, const uint8_t* secret_keys, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLKEM1024Decaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM1024Decaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM1024Decaps::ciphertext_size;
    auto secret_key    = secret_keys + block * MLKEM1024Decaps::secret_key_size;
    auto work          = workspace   + block * MLKEM1024Decaps::workspace_size;

    MLKEM1024Decaps().execute(shared_secret, ciphertext, secret_key, work, smem_ptr);
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
        printf("🚀 ML-KEM-1024 GH200 Maximum Performance Benchmarking\n");
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

    std::vector<unsigned int> generateOptimalBatchSizes() {
        size_t single_op_memory = MLKEM1024Key::public_key_size + MLKEM1024Key::secret_key_size + 
                                 MLKEM1024Key::workspace_size + MLKEM1024Key::entropy_size;
        
        size_t usable_memory = static_cast<size_t>(gpu_info.free_memory * 0.75);
        unsigned int max_memory_batch = usable_memory / single_op_memory;
        
        unsigned int REASONABLE_MAX = 500000;
        max_memory_batch = std::min(max_memory_batch, REASONABLE_MAX);
        
        unsigned int sm_4x = gpu_info.sm_count * 4;
        unsigned int sm_8x = gpu_info.sm_count * 8;
        unsigned int sm_16x = gpu_info.sm_count * 16;
        
        std::vector<unsigned int> batch_sizes = {
            10000, 25000, 50000, 100000, 200000,
            sm_4x, sm_8x, sm_16x,
            max_memory_batch / 4, max_memory_batch / 2,
            static_cast<unsigned int>(max_memory_batch * 0.75),
            max_memory_batch
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
        
        for (int i = 0; i < 10; i++) {
            kernel_func();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
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

// Safe ML-KEM-1024 functions
bool ml_kem_keygen_safe(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, 
                       unsigned int batch, int iterations, MaxPerformanceBenchmark& benchmark) {
    
    auto length_public_key = MLKEM1024Key::public_key_size;
    auto length_secret_key = MLKEM1024Key::secret_key_size;

    auto workspace = make_workspace<MLKEM1024Key>(batch);
    auto randombytes = get_entropy<MLKEM1024Key>(batch);

    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation) {
        if (d_public_key) cudaFree(d_public_key);
        if (d_secret_key) cudaFree(d_secret_key);
        destroy_workspace(workspace);
        release_entropy(randombytes);
        return false;
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);

    size_t memory_ops = (length_public_key + length_secret_key + 
                        MLKEM1024Key::workspace_size + MLKEM1024Key::entropy_size);

    auto kernel_func = [&]() {
        keygen_kernel<<<batch, MLKEM1024Key::BlockDim>>>(d_public_key, d_secret_key, workspace, randombytes);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Key Generation", kernel_func, batch, iterations, memory_ops);

    CUDA_CHECK(cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_public_key));
    CUDA_CHECK(cudaFree(d_secret_key));
    destroy_workspace(workspace);
    release_entropy(randombytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return true;
}

bool ml_kem_encaps_safe(std::vector<uint8_t>& ciphertexts, std::vector<uint8_t>& shared_secrets,
                       const std::vector<uint8_t>& public_keys, unsigned int batch, int iterations, 
                       MaxPerformanceBenchmark& benchmark) {
    
    auto length_ciphertext = MLKEM1024Encaps::ciphertext_size;
    auto length_sharedsecret = MLKEM1024Encaps::shared_secret_size;
    auto length_public_key = MLKEM1024Encaps::public_key_size;

    auto workspace = make_workspace<MLKEM1024Encaps>(batch);
    auto randombytes = get_entropy<MLKEM1024Encaps>(batch);

    uint8_t* d_ciphertext = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_public_key = nullptr;

    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaError_t error3 = cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation || error3 == cudaErrorMemoryAllocation) {
        if (d_ciphertext) cudaFree(d_ciphertext);
        if (d_sharedsecret) cudaFree(d_sharedsecret);
        if (d_public_key) cudaFree(d_public_key);
        destroy_workspace(workspace);
        release_entropy(randombytes);
        return false;
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);
    CUDA_CHECK(error3);

    CUDA_CHECK(cudaMemcpy(d_public_key, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice));

    size_t memory_ops = (length_ciphertext + length_sharedsecret + length_public_key + 
                        MLKEM1024Encaps::workspace_size + MLKEM1024Encaps::entropy_size);

    auto kernel_func = [&]() {
        encaps_kernel<<<batch, MLKEM1024Encaps::BlockDim>>>(d_ciphertext, d_sharedsecret, d_public_key, workspace, randombytes);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Encapsulation", kernel_func, batch, iterations, memory_ops);

    CUDA_CHECK(cudaMemcpy(ciphertexts.data(), d_ciphertext, length_ciphertext * batch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ciphertext));
    CUDA_CHECK(cudaFree(d_sharedsecret));
    CUDA_CHECK(cudaFree(d_public_key));
    destroy_workspace(workspace);
    release_entropy(randombytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return true;
}

bool ml_kem_decaps_safe(std::vector<uint8_t>& shared_secrets, const std::vector<uint8_t>& ciphertexts,
                       const std::vector<uint8_t>& secret_keys, unsigned int batch, int iterations,
                       MaxPerformanceBenchmark& benchmark) {
    
    auto length_ciphertext = MLKEM1024Decaps::ciphertext_size;
    auto length_sharedsecret = MLKEM1024Decaps::shared_secret_size;
    auto length_secret_key = MLKEM1024Decaps::secret_key_size;

    auto workspace = make_workspace<MLKEM1024Decaps>(batch);

    uint8_t* d_ciphertext = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaError_t error1 = cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaError_t error2 = cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaError_t error3 = cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);
    
    if (error1 == cudaErrorMemoryAllocation || error2 == cudaErrorMemoryAllocation || error3 == cudaErrorMemoryAllocation) {
        if (d_ciphertext) cudaFree(d_ciphertext);
        if (d_sharedsecret) cudaFree(d_sharedsecret);
        if (d_secret_key) cudaFree(d_secret_key);
        destroy_workspace(workspace);
        return false;
    }
    
    CUDA_CHECK(error1);
    CUDA_CHECK(error2);
    CUDA_CHECK(error3);

    CUDA_CHECK(cudaMemcpy(d_ciphertext, ciphertexts.data(), length_ciphertext * batch, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_secret_key, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice));

    size_t memory_ops = (length_ciphertext + length_sharedsecret + length_secret_key + 
                        MLKEM1024Decaps::workspace_size);

    auto kernel_func = [&]() {
        decaps_kernel<<<batch, MLKEM1024Decaps::BlockDim>>>(d_sharedsecret, d_ciphertext, d_secret_key, workspace);
        CUDA_CHECK(cudaGetLastError());
    };

    benchmark.runSilentBenchmark("Decapsulation", kernel_func, batch, iterations, memory_ops);

    CUDA_CHECK(cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ciphertext));
    CUDA_CHECK(cudaFree(d_sharedsecret));
    CUDA_CHECK(cudaFree(d_secret_key));
    destroy_workspace(workspace);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return true;
}

void printFinalResults(const std::vector<BenchmarkResult>& results) {
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                               🏆 GH200 ML-KEM-1024 MAXIMUM PERFORMANCE RESULTS 🏆              ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-15s ║ %-10s ║ %-12s ║ %-15s ║ %-12s ║ %-10s ║ %-8s ║\n", 
           "Operation", "Batch", "Time (ms)", "Throughput", "Bandwidth", "GPU Util%", "Mem(GB)");
    printf("║                 ║            ║              ║    (ops/sec)    ║    (GB/s)    ║            ║          ║\n");
    printf("╠═════════════════╬════════════╬══════════════╬═════════════════╬══════════════╬════════════╬══════════╣\n");
    
    std::map<std::string, BenchmarkResult> best_results;
    for (const auto& result : results) {
        if (best_results.find(result.operation_name) == best_results.end() || 
            result.throughput_ops_per_sec > best_results[result.operation_name].throughput_ops_per_sec) {
            best_results[result.operation_name] = result;
        }
    }
    
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
    auto batch_sizes = benchmark.generateOptimalBatchSizes();
    
    int iterations = 100;
    
    printf("Testing %zu batch configurations... ", batch_sizes.size());
    fflush(stdout);
    
    for (size_t i = 0; i < batch_sizes.size(); i++) {
        auto batch = batch_sizes[i];
        
        printf(".");
        fflush(stdout);
        
        std::vector<uint8_t> public_keys(MLKEM1024Key::public_key_size * batch);
        std::vector<uint8_t> secret_keys(MLKEM1024Key::secret_key_size * batch);
        std::vector<uint8_t> ciphertexts(MLKEM1024Encaps::ciphertext_size * batch);
        std::vector<uint8_t> encaps_shared_secrets(MLKEM1024Encaps::shared_secret_size * batch);
        std::vector<uint8_t> decaps_shared_secrets(MLKEM1024Decaps::shared_secret_size * batch);
        
        bool keygen_success = ml_kem_keygen_safe(public_keys, secret_keys, batch, iterations, benchmark);
        if (!keygen_success) {
            printf("⚠");
            continue;
        }
        
        bool encaps_success = ml_kem_encaps_safe(ciphertexts, encaps_shared_secrets, public_keys, batch, iterations, benchmark);
        if (!encaps_success) {
            printf("⚠"); 
            continue;
        }
        
        bool decaps_success = ml_kem_decaps_safe(decaps_shared_secrets, ciphertexts, secret_keys, batch, iterations, benchmark);
        if (!decaps_success) {
            printf("⚠");
            continue;
        }
    }
    
    printf(" Done!\n\n");
    
    printFinalResults(benchmark.getResults());
    
    printf("🎉 ML-KEM-1024 Benchmarking completed! Your GH200 has been pushed to its limits! 💪\n");
    return 0;
}

