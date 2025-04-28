// naive_matmul_with_timing.cu
#include <iostream>
#include <cuda_runtime.h>

//#define N 512  // Size of the matrix (N x N)
#define EPSILON 1e-3  // For correctness check

// CUDA Kernel for naive matrix multiplication
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CPU reference matmul
void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// Function to check if two matrices are approximately equal
bool verify_result(const float* a, const float* b, int n) {
    for (int i = 0; i < n * n; ++i) {
        if (fabs(a[i] - b[i]) > EPSILON) {
            std::cout << "Mismatch at index " << i 
                      << ": GPU value = " << a[i]
                      << ", CPU value = " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N = 1<<10;
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_ref = (float*)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // Or random values
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Create CUDA events
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    // Host to Device copy timing
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    // Kernel launch timing
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    cudaEventRecord(start_kernel);
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_kernel);

    // Device to Host copy timing
    cudaEventRecord(start_d2h);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);

    // Wait for all events to complete
    cudaEventSynchronize(stop_h2d);
    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop_d2h);

    // Calculate elapsed times
    float time_h2d, time_kernel, time_d2h;
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    std::cout << "H2D Copy Time (ms): " << time_h2d << std::endl;
    std::cout << "Kernel Execution Time (ms): " << time_kernel << std::endl;
    std::cout << "D2H Copy Time (ms): " << time_d2h << std::endl;

    // Compute CPU reference result
    matmul_cpu(h_A, h_B, h_C_ref, N);

    // Verify correctness
    if (verify_result(h_C, h_C_ref, N)) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is incorrect!" << std::endl;
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    // Destroy events
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
