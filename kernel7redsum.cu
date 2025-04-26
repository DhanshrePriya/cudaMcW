#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
 
using std::accumulate;
using std::cout;
using std::vector;
 
#define SHMEM_SIZE 256
 
 
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
 
template<int blocksize>
__device__ void warpReduce(volatile int* shmem_ptr, unsigned int t) {
    // printf(" %d ", t);
    if(blocksize >= 64) shmem_ptr[t] += shmem_ptr[t + 32];
    if(blocksize >= 32) shmem_ptr[t] += shmem_ptr[t + 16];
    if(blocksize >= 16) shmem_ptr[t] += shmem_ptr[t + 8];
    if(blocksize >= 8) shmem_ptr[t] += shmem_ptr[t + 4];
    if(blocksize >= 4) shmem_ptr[t] += shmem_ptr[t + 2];
    if(blocksize >= 2) shmem_ptr[t] += shmem_ptr[t + 1];
}
 
template<unsigned int blocksize>
__global__ void sumReduction(int *v, int N) {
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];
 
    // Calculate thread ID
    unsigned tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blocksize*2) + threadIdx.x;
    unsigned int gridSize = blocksize*2*gridDim.x;
 
    partial_sum[tid] = 0;
    // Load elements into shared memory
 
    // grid size is 64 times smaller compared to N.
    // we are doing 2 loads per thread single time so grid size will be 32 times smaller
    // so this loop will make all elements increasing work per thread.
    while(i < N) {  // this loop always run 32 times ex: for t0 in b0 0+(131072*32) = 1 << 22
        partial_sum[tid] += v[i] + ((i + blockDim.x < N) ? v[i + blockDim.x] : 0);  
        i += gridSize;
    }
    __syncthreads();
 
    if(blocksize >= 512) {
        if(tid < 256) {
            partial_sum[tid] += partial_sum[tid + 256];
        }
        __syncthreads();
    }
    if(blocksize >= 256) {
        if(tid < 128) {
            partial_sum[tid] += partial_sum[tid + 128];
        }
        __syncthreads();
    }
    if(blocksize >= 128) {
        if(tid < 64) {
            partial_sum[tid] += partial_sum[tid + 64];
        }
        __syncthreads();
    }
 
    if(tid < 32) {
        warpReduce<blocksize>(partial_sum, tid);
    }
 
    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (tid == 0) {
        // printf("%d\n", partial_sum[0]);
        v[blockIdx.x] = partial_sum[0];
        // atomicAdd(&v_r[blockIdx.x], partial_sum[0]);
    }
 
}
 
int main() {
    // Vector size
    int N = 1 << 10;
    size_t bytes = N * sizeof(int);
 
    // Host data
    vector<int> h_v(N);
    vector<int> h_v_r(N);
 
    // Initialize the input data
    for(int i=0;i<h_v.size();i++) h_v[i] = 1;
 
 
    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
   
    cudaEventRecord(start);
 
    // Allocate device memory
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);
   
    // Copy to device
    cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);
   
    // TB Size
    const int TB_SIZE = 256;
 
    // Grid Size
    int GRID_SIZE = (N/TB_SIZE)/256;
    //int GRID_SIZE = (int)ceil((N + TB_SIZE -1) / TB_SIZE/2);
 
    cudaEventRecord(kernelStart);
 
    // Launch kernel
    while(GRID_SIZE > 1) {
        sumReduction<256><<<GRID_SIZE, TB_SIZE>>>(d_v, N);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        N = GRID_SIZE;
        GRID_SIZE /= TB_SIZE;
    }
    // cout << "Final Kernel\n";
    sumReduction<256><<<1, TB_SIZE>>>(d_v, N);
 
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
 
    // Copy to host;
    cudaMemcpy(h_v_r.data(), d_v, bytes, cudaMemcpyDeviceToHost);
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    cudaEventElapsedTime(&time, start, stop);
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    std::cout << "GPU All Kernels time: " << kernelTime << " ms" << std::endl;
    std::cout << "GPU Total time (H2D + All Kernels + D2H): " << time << " ms" << std::endl;
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
 
 
    // Print the result
    // cout << h_v_r[0] << " " << h_v_r[1] << "\n";
    // cout << std::accumulate(begin(h_v), end(h_v), 0);
    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
 
    cout << "COMPLETED SUCCESSFULLY\n";
 
    return 0;
}
 