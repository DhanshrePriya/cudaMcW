#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define KERNEL_SIZE 3  
#define TILE_SIZE 16
__global__ void Conv2D(float* input, float* kernel, float* output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int half_k = KERNEL_SIZE / 2;
    
    __shared__ float k[KERNEL_SIZE][KERNEL_SIZE]; 
    __shared__ float in[TILE_SIZE + 2][TILE_SIZE + 2]; 

    if (threadIdx.y < KERNEL_SIZE && threadIdx.x < KERNEL_SIZE) {
        k[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * KERNEL_SIZE + threadIdx.x];
    }

    if (row < height && col < width) {
        in[threadIdx.y + 1][threadIdx.x + 1] = input[row * width + col];
    }

    __syncthreads();  

    if (row < half_k || row >= height - half_k || col < half_k || col >= width - half_k) {
        output[row * width + col] = input[row * width + col];
    }

    float sum = 0.0f;
    for (int i = -half_k; i <= half_k; i++) {
        for (int j = -half_k; j <= half_k; j++) {
            sum += in[threadIdx.y + 1 + i][threadIdx.x + 1 + j] * k[i + half_k][j + half_k];
        }
    }

    output[row * width + col] = sum;
}

void ConvolutionGPU(int width, int height, float* input, float* kernel, float* output) {
    float *d_input, *d_kernel, *d_output;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    Conv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, end);
    std::cout << "TIME: " << gpu_time << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    int width = 1024, height = 1024;
    std::vector<float> input(width * height, 1.234f);
    std::vector<float> kernel = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };  
    std::vector<float> output(width * height, 0.0f);

    ConvolutionGPU(width, height, input.data(), kernel.data(), output.data());

    return 0;
}
