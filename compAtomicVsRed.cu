#include <iostream>
#include <vector>
#include <cuda.h>
#include <cstdlib>
#include <ctime>

using namespace std;

#define N 1048576  
#define THREADS_PER_BLOCK 1024

__global__ void sum_atomic(float* d_in, float* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(d_out, d_in[idx]);
    }
}

__global__ void sum_reduction(float* d_in, float* d_out) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

int main() {
    vector<float> h_in(N);
    float *d_in, *d_out_atomic, *d_out_reduction;
    float sum_atomic_result = 0.0f, sum_reduction_result = 0.0f;

    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(rand() % 100 + 1); 
    }

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out_atomic, sizeof(float));
    cudaMalloc(&d_out_reduction, sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out_atomic, 0, sizeof(float));
    cudaMemset(d_out_reduction, 0, sizeof(float));

    cudaEvent_t start, stop;
    float time_atomic, time_reduction;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sum_atomic<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_in, d_out_atomic);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_atomic, start, stop);

    cudaEventRecord(start);
    sum_reduction<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_in, d_out_reduction);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_reduction, start, stop);

 
    cudaMemcpy(&sum_atomic_result, d_out_atomic, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_reduction_result, d_out_reduction, sizeof(float), cudaMemcpyDeviceToHost);


    cout << "Sum (atomic): " << sum_atomic_result << " | Time: " << time_atomic << " ms\n";
    cout << "Sum (reduction): " << sum_reduction_result << " | Time: " << time_reduction << " ms\n";

    cudaFree(d_in);
    cudaFree(d_out_atomic);
    cudaFree(d_out_reduction);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
