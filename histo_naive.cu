#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#define NUM_BINS 256

__global__ void kernel(float *d_image, int *d_hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&d_hist[(int)d_image[idx]], 1); 
    }
}

void cpuHistogram(float *h_image, int *h_hist, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        h_hist[(int)h_image[i]]++;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "CPU time: " << time << " ms" << std::endl;
}

int main() {
    // img and hist size in cpu
    const int imgSize = 1 << 20; 
    size_t imgBytes = imgSize * sizeof(float);
    size_t histBytes = NUM_BINS * sizeof(int);

    // img and hist in cpu 
    float *h_image = new float[imgSize];
    int *h_hist = new int[NUM_BINS]();  
    int *h_hist_gpu = new int[NUM_BINS]();  

    // random pixel initialization
    for (int i = 0; i < imgSize; i++) {
        h_image[i] = rand() % NUM_BINS;
    }

    // mem allocation in gpu
    float *d_image;
    int *d_hist;
    cudaMalloc(&d_image, imgBytes);
    cudaMalloc(&d_hist, histBytes);
    cudaMemset(d_hist, 0, histBytes);

    // cuda events creation for diferent instances
    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    // (H2D) timing
    cudaEventRecord(startH2D);
    cudaMemcpy(d_image, h_image, imgBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);

    //kernel
    int threads = 512;
    int blocks = (imgSize + threads - 1) / threads;

    cudaEventRecord(startKernel);
    kernel<<<blocks, threads>>>(d_image, d_hist, imgSize);
    cudaEventRecord(stopKernel);
    cudaDeviceSynchronize();

    //(D2H) timing
    cudaEventRecord(startD2H);
    cudaMemcpy(h_hist_gpu, d_hist, histBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);

    // cpu validation func call
    cpuHistogram(h_image, h_hist, imgSize);

    //assert values
    for (int i = 0; i < NUM_BINS; i++) {
        assert(h_hist[i] == h_hist_gpu[i]);
    }
    std::cout << "Histogram verification correct !" << std::endl;

    float timeH2D, timeKernel, timeD2H;
    cudaEventSynchronize(stopH2D);
    cudaEventSynchronize(stopKernel);
    cudaEventSynchronize(stopD2H);

    cudaEventElapsedTime(&timeH2D, startH2D, stopH2D);
    cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
    cudaEventElapsedTime(&timeD2H, startD2H, stopD2H);

    std::cout << "H2D Time: " << timeH2D << " ms" << std::endl;
    std::cout << "Kernel Time: " << timeKernel << " ms" << std::endl;
    std::cout << "D2H Time: " << timeD2H << " ms" << std::endl;
    std::cout << "Total time: " << timeH2D + timeKernel + timeD2H << " ms" << std::endl;

    // free mem 
    delete[] h_image;
    delete[] h_hist;
    delete[] h_hist_gpu;
    cudaFree(d_image);
    cudaFree(d_hist);

    // kill events
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);

    return 0;
}
