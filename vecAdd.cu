#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 256

// Naïve GPU Kernel (1 thread per element)
__global__ void vector_add_naive(int *A, int *B, int *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// Optimized GPU Kernel using Brent's Theorem (1 thread per log N elements)
__global__ void vector_add_optimized(int *A, int *B, int *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Threads work in chunks

    for (int i = tid; i < N; i += stride) {  
        C[i] = A[i] + B[i];  
    }
}

// CPU Version for Benchmarking
// void vector_add_cpu(int *A, int *B, int *C, int N) {
//     for (int i = 0; i < N; i++) {
//         C[i] = A[i] + B[i];
//     }
// }

// Helper function to measure GPU execution time
float measure_gpu_time(void (*kernel)(int*, int*, int*, int), int *A, int *B, int *C, int N) {
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    kernel<<<grid_size, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return gpu_time;
}

int main() {
    int N = 1 << 30; // 1 Million Elements
    vector<int> A(N), B(N), C(N);

    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // CPU Timing
    // auto cpu_start = high_resolution_clock::now();
    // vector_add_cpu(A.data(), B.data(), C.data(), N);
    // auto cpu_end = high_resolution_clock::now();
    // double cpu_time = duration<double, milli>(cpu_end - cpu_start).count();
    
    // GPU Naïve
    float gpu_time_naive = measure_gpu_time(vector_add_naive, A.data(), B.data(), C.data(), N);

    // GPU Optimized (Brent’s Theorem)
    float gpu_time_optimized = measure_gpu_time(vector_add_optimized, A.data(), B.data(), C.data(), N);

    //cout << "CPU Execution Time: " << cpu_time << " ms" << endl;
    cout << "GPU Naïve Execution Time: " << gpu_time_naive << " ms" << endl;
    cout << "GPU Optimized Execution Time: " << gpu_time_optimized << " ms" << endl;

    return 0;
}
