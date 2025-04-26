#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void HelloWorld(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Hello from thread %d\n", idx);
    }
}

int main() {
    int n = 10;
    int blocks = 2;
    int threads = 5;

    // Launch kernel
    HelloWorld<<<blocks, threads>>>(n);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return 0;
}
