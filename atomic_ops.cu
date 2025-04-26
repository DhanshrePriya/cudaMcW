#include <stdio.h>

__global__ void atomic_shared_kernel(int *output) {
    __shared__ int sharedVar;  
    if (threadIdx.x == 0) 
        sharedVar = 0;  

    
    sharedVar += 1; // op = 1

    //atomicAdd(&sharedVar, 1);  // op = 10

    __syncthreads();  

    if (threadIdx.x == 0) 
        *output = sharedVar;  
}

int main() {
    int h_result = 0, *d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    atomic_shared_kernel<<<1, 10>>>(d_result);  

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Final shared memory value: %d\n", h_result);  

    cudaFree(d_result);
    return 0;
}

