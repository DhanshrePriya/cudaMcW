#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
#define BLOCK_SIZE 16

__global__ void MatrixTranKernel(int* A, int* AT, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < n && c < n) {
            AT[r * n + c] = A[c * n + r];  
    }

}
void TransposeGPU(int* A, int* AT, int n) {
    int *d_A, *d_AT;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaEventRecord(start);

    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_AT, n * n * sizeof(int));
    cudaMemcpy(d_A, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    
  
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);    

    MatrixTranKernel<<<grid, block>>>(d_A, d_AT, n);
    cudaError_t error = cudaGetLastError();
 
if (error != cudaSuccess) {
 
    printf("CUDA error 11: %s\n", cudaGetErrorString(error));
}
    cudaDeviceSynchronize();
    error = cudaGetLastError();
 
if (error != cudaSuccess) {
 
    printf("CUDA error 22: %s\n", cudaGetErrorString(error));
}

    cudaMemcpy(AT, d_AT, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cout << "BLOCK_SIZE: " << BLOCK_SIZE << endl;
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, end);
    cout << "GPU Time: " << gpu_time << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_AT);

}

int main() {
    int n = 1 << 14;
    std::vector<int> A(n * n);
    std::vector<int> AT(n * n, 0);

    for(int i = 0; i < n * n; i++) {
        A[i] = i;
    }
    TransposeGPU(A.data(), AT.data(), n);


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(AT[j * n + i] == A[i * n + j]); 
        }
    }
    cout << "CORRECT"<< endl;
    return 0;
}
