#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

#define TILE_SIZE 32  
#define WPT 8    // Increased Work Per Thread
#define VEC_SIZE 4  // Vectorized load (float4)
#define PAD 1  // Shared memory padding to avoid bank conflicts

__global__ void MatrixMulKernel(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int M, int N, int K) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE + PAD];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx * WPT;

    float acc[WPT] = {0.0f};  // 2D Register Blocking

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K)
            Asub[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            Asub[ty][tx] = 0.0f;

        // Load B using vectorized loads
        if ((t * TILE_SIZE + ty) < K && col < N) {
            float4 b4_1 = *reinterpret_cast<const float4*>(&B[(t * TILE_SIZE + ty) * N + col]);
            float4 b4_2 = *reinterpret_cast<const float4*>(&B[(t * TILE_SIZE + ty) * N + col + 4]);
            Bsub[ty][tx * WPT] = b4_1.x;
            Bsub[ty][tx * WPT + 1] = b4_1.y;
            Bsub[ty][tx * WPT + 2] = b4_1.z;
            Bsub[ty][tx * WPT + 3] = b4_1.w;
            Bsub[ty][tx * WPT + 4] = b4_2.x;
            Bsub[ty][tx * WPT + 5] = b4_2.y;
            Bsub[ty][tx * WPT + 6] = b4_2.z;
            Bsub[ty][tx * WPT + 7] = b4_2.w;
        } else {
            for (int i = 0; i < WPT; i++)
                Bsub[ty][tx * WPT + i] = 0.0f;
        }
        __syncthreads();

        // Compute using 2D register blocking with FMA
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < WPT; i++) {
                acc[i] = fmaf(Asub[ty][k], Bsub[k][tx * WPT + i], acc[i]);
            }
        }
        __syncthreads();
    }

    // Store results back to C
    if (row < M && col < N) {
        for (int i = 0; i < WPT; i++) {
            C[row * N + col + i] = acc[i];
        }
    }
}

void MatrixMulGPU(int M, int N, int K, float* A, float* B, float* C) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE / WPT, TILE_SIZE);  // Increased WPT
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    MatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int m = 1 << 10, n = 1 << 10, k = 1 << 10;

    vector<float> A(m * k);
    vector<float> B(k * n);
    vector<float> C(m * n);

    for(int i = 0; i < m * k; i++)
        A[i] = rand() % 100;

    for(int i = 0; i < k * n; i++)
        B[i] = rand() % 100;

    MatrixMulGPU(m, n, k, A.data(), B.data(), C.data());

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0;
            for(int q = 0; q < k; q++)
                sum += A[i * k + q] * B[q * n + j];
            assert(abs(sum - C[i * n + j]) < 1e-3);
        }
    }
    cout << "CORRECT!!!" << endl;
    return 0;
}
