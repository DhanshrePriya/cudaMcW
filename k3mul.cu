#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 16  
#define WPT 8 

__global__ void matMulSharedWPT(float *C, const float *A, const float *B, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty * WPT;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum[WPT] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        for (int i = 0; i < WPT; ++i) {
            int loadRow = row + i;
            As[ty * WPT + i][tx] = (loadRow < M && t * TILE_SIZE + tx < K) 
                ? A[loadRow * K + t * TILE_SIZE + tx] : 0.0f;

            Bs[ty * WPT + i][tx] = (t * TILE_SIZE + ty * WPT + i < K && col < N) 
                ? B[(t * TILE_SIZE + ty * WPT + i) * N + col] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int i = 0; i < WPT; ++i) {
                sum[i] += As[ty * WPT + i][k] * Bs[k][tx];
            }
        }
        
        __syncthreads();
    }

    for (int i = 0; i < WPT; ++i) {
        if (row + i < M && col < N)
            C[(row + i) * N + col] = sum[i];
    }
}

void cpuMatMul(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 1024, N = 512, K = 1024;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, sizeA);
    cudaMallocManaged(&B, sizeB);
    cudaMallocManaged(&C, sizeC);

    std::vector<float> A_host(M * K), B_host(K * N), C_cpu(M * N);
    for (int i = 0; i < M * K; i++) A_host[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < K * N; i++) B_host[i] = static_cast<float>(rand() % 10);
    std::copy(A_host.begin(), A_host.end(), A);
    std::copy(B_host.begin(), B_host.end(), B);

    dim3 blockDim(TILE_SIZE, TILE_SIZE / WPT);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulSharedWPT<<<gridDim, blockDim>>>(C, A, B, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms\n";

    cpuMatMul(C_cpu, A_host, B_host, M, N, K);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cpu[i] - C[i]) > 1e-3) {
            correct = false;
            break;
        }
    }
    std::cout << (correct ? "Correct\n" : "Aborted\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
