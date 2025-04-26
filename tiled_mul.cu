#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

#define TILE_SIZE 16 
#define BLOCK_SIZE 16   

__global__ void MatrixMulKernelTiled(float* A, float* B, float* C, int m, int n, int k) {
    
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];  
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (n + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        
        if (row < m && tileIdx * TILE_SIZE + threadIdx.x < n)
            Asub[threadIdx.y][threadIdx.x] = A[row * n + tileIdx * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < k && tileIdx * TILE_SIZE + threadIdx.y < n)
            Bsub[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * k + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();  
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
}

void MatrixMulGPU(int m, int n, int k, float* A, float* B, float* C) {
    float *d_A, *d_B, *d_C;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMalloc(&d_C, m * k * sizeof(float));

    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatrixMulKernelTiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, end);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int m = 1024, n = 1024, k = 1024;

    std::vector<float> A(m * n, rand() % 10);
    std::vector<float> B(n * k, rand() % 10);
    std::vector<float> C(m * k, 0.0f);

    MatrixMulGPU(m, n, k, A.data(), B.data(), C.data());

    return 0;
}
