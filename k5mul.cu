#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
 
using namespace std;
 
#define TSM 32    // Tile-size in dimension M
#define TSN 32    // Tile-size in dimension N
#define TSK 32    // Tile-size in dimension K
#define WPTN 8    // Work-per-thread in dimension N
#define WPT 1
#define RTSN (TSN/WPTN) // Reduced tile-size in N
#define LPT ((TSK*TSM)/(TSM*RTSN)) // Loads-per-thread per tile
 
// For Transpose kernel
#define THREADSX 4
#define THREADSY 4
#define THREADS 4
#define TS 4
 
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
 
// Simple transpose kernel for a P * Q matrix
__global__ void transpose(const int M, const int N, const  float* input, float* output) {
 
    // Thread identifiers
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ID0 = blockIdx.x*THREADSX + tx; // 0..N
    const int ID1 = blockIdx.y*THREADSY + ty; // 0..M
 
    // Set-up the local memory for shuffling
    __shared__ float buffer[THREADSX][THREADSY];
 
    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < N && ID1 < M) {
        // printf("%d ", input[ID1*P + ID0]);
        buffer[ty][tx] = input[ID1*N + ID0];
    }
 
    // Synchronise all threads
    __syncthreads();
 
    if (ID0 < M && ID1 < N) {
        output[ID0 * M + ID1] = buffer[ty][tx];
    }
    // Store the transposed result (coalesced)
}
 
 
__global__ void myGEMM5(const int M, const int N, const int K, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x * WPT + threadIdx.x;
    // printf("h");
    // Statically allocated shared memory
    __shared__ float s_a[THREADS][THREADS];
    __shared__ float s_b[THREADS][THREADS];
    __shared__ int indexCalc;
 
    const unsigned int RTS = THREADS/WPT;
    // const unsigned int RTS = blockDim.x*gridDim.x;
    // int tmp = 0;
    float acc[WPT];
    for (int i = 0; i < WPT; i++)
    {
        acc[i] = 0.0f;
    }
   
    if(threadIdx.y == 0 && threadIdx.x == 0) indexCalc = col*K;
    __syncthreads();
    // Sweep tile across Matrix
    for(int i = 0; i < K; i+=(THREADS)) {
        // Load in elements for this tile
        // printf("h");
        for(int w = 0; w < WPT; w++) {
           
            s_a[threadIdx.y][threadIdx.x + w * RTS] = a[row * K + i + threadIdx.x + w * RTS];
            // s_b[threadIdx.y][threadIdx.x + w * RTS] = b[row * K + i + threadIdx.x + w * RTS];
            s_b[threadIdx.y][threadIdx.x + w * RTS] = b[indexCalc + threadIdx.y * K + i + threadIdx.x + w * RTS];
            // if(row == 4 && col == 5) {
            //   // printf("%d %d %d %d %d\n", r, threadIdx.y * K, i, threadIdx.x, w * RTS);
            // }
        }
        __syncthreads();
        // if(row == 4 && col == 4) {
        //     printf("\n\n\n");
        //   for(int i=0;i<TS;i++) {
        //     for(int j=0;j<TS;j++) {
        //       printf("%f ", s_a[i][j]);
        //     }
        //     printf("\n");
        //   }
        //   printf("\n");
        //   for(int i=0;i<TS;i++) {
        //     for(int j=0;j<TS;j++) {
        //       printf("%f ", s_b[i][j]);
        //     }
        //     printf("\n");
        //   }
        // }
        // The below line should be removed
        // __syncthreads();
        for(int j = 0; j < (THREADS); j++) {
            for(int w = 0; w < WPT; w++) {
                acc[w] += s_a[threadIdx.y][j] * s_b[threadIdx.x][j];
                // if(row == 0 && col == 0)
                //   printf("\n%f*%f=%f\n",s_a[threadIdx.y][j], s_b[threadIdx.x][j],acc[w]);
            }
        }
        __syncthreads();
    }
    for(int w = 0; w < WPT; w++) {
        if (row < M && col < N){
            c[row * N + col + w * RTS] = acc[w];
        }
    }
 
}

// Check result on the CPU
void verify_result(vector<float> &a, vector<float> &b, vector<float> &c, int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    cout << "Matrix C (CPU) \n";
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        // For every element in the row-column pair
        int tmp = 0;
        for (int k = 0; k < N; k++) {
          // Accumulate the partial results
          tmp += a[i * N + k] * b[k * K + j];
        }
        // Check against the CPU result
        cout << tmp << " ";
        if(tmp != c[i * K + j]) {
          cout << i << " " << j << "\n";
        }
        // assert(tmp == c[i * K + j]);
      }
      // cout << "\n";
    }
    // Calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
}
 
int main() {
    // Matrix dimensions
    int M = 1 << 6;
    int N = 1 << 6;
    int K = 1 << 6;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
 
    vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0), h_BT(N * K);
    for (float i = 0; i < M * K; i++) {
        // h_A[i] = rand() % 100;
        h_A[i] = i;
    }
    for (float i = 0; i < K * N; i++) {
        h_B[i] = rand() % 100;
        h_B[i] = i;
    }
   
  cout << "Matrix A: " << M << "x" << N << "\n";
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      cout << h_A[i*N + j] << " ";
    }
    cout << "\n";
  }
 
  cout << "Matrix B: " << N << "x" << K << "\n";
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < K; j++) {
      cout << h_B[i*K + j] << " ";
    }
    cout << "\n";
  }
 
 
    float *d_A, *d_B, *d_BT, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_BT, bytesB);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
 
    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT, h_B.data(), bytesB, cudaMemcpyHostToDevice);
 
    dim3 threadsT(THREADSY, THREADSX);
    dim3 blocksT(N/THREADSY, K/THREADSX);
    transpose<<<blocksT, threadsT>>>(M, N, d_BT, d_B);
    cudaMemcpy(h_BT.data(), d_B, bytesB, cudaMemcpyDeviceToHost);
    cout << "Tranpose B: " << K << "x" << N << "\n";
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < K; j++) {
        cout << h_BT[i*K + j] << " ";
      }
      cout << "\n";
    }
    // dim3 threads(TSM, RTSN);
    // dim3 blocks(M / TSM, N / TSN);
 
    dim3 threads(THREADS/WPT, THREADS);
    dim3 blocks(N/THREADS, M/THREADS);
 
 
    myGEMM5<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
   
    // cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
 
    cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost);
    cout << "Matrix C(GPU): " << M << "x" << N << "\n";
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
        cout << h_C[i*K + j] << " ";
        }
        cout << "\n";
    }
    verify_result(h_A, h_B, h_C, M, K, N);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
 
 