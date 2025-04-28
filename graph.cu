#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

__global__ void incrementKernel(int *data, int increment) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += increment;
}

int main() {
    int *d_data;
    int h_data[1024] = {0}; 
    cudaMalloc(&d_data, 1024 * sizeof(int));
    cudaMemcpy(d_data, h_data, 1024 * sizeof(int), cudaMemcpyHostToDevice);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 1: Create CUDA Graph
    cudaGraphCreate(&graph, 0);

    // Step 2: Add Kernel Node to the Graph
    cudaGraphNode_t kernelNode;
    int increment = 1;  // Initial increment
    void *kernelArgs[] = { &d_data, &increment };
    cudaKernelNodeParams kernelParams = {0};

    kernelParams.func = (void*)incrementKernel;
    kernelParams.gridDim = dim3(1);
    kernelParams.blockDim = dim3(1024);
    kernelParams.sharedMemBytes = 0; 
    kernelParams.kernelParams = kernelArgs;
    
    cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams);

    // Step 3: Instantiate the Graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Step 4: Execute and Dynamically Update the Graph
    for (int i = 1; i <= 5; i++) {  
        increment = i;  // Update the increment value

        // Modify the kernel parameters with the new increment
        cudaKernelNodeParams newParams = kernelParams;
        newParams.kernelParams = kernelArgs;  // Update kernel arguments

        // Apply the update to the existing graph execution instance
        cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);

        // Launch the updated graph
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        // Copy data back to host and print first 10 elements
        cudaMemcpy(h_data, d_data, 1024 * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Iteration " << i << ": ";
        for (int j = 0; j < 10; j++) {
            std::cout << h_data[j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    return 0;
}
