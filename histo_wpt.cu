#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>

#define NUM_BINS 256
#define SIZE 1 << 24
#define TILE 512

__global__ void kernel(const unsigned char *d_image, int *d_hist, int wpt)
{
    int idx = blockIdx.x * blockDim.x * wpt + threadIdx.x;
    __shared__ int hist[NUM_BINS];
    if (threadIdx.x < NUM_BINS)
    {
        hist[threadIdx.x] = 0;
        //__syncthreads();
    }
    int i = 0;
    while (i < wpt)
    {
        if (idx + blockDim.x * i < SIZE)
            atomicAdd(&hist[d_image[idx + blockDim.x * i]], 1);
        i += 1;
    }
    __syncthreads();
    if (threadIdx.x < NUM_BINS)
        atomicAdd(&d_hist[threadIdx.x], hist[threadIdx.x]);
}

// CPU Histogram
void cpuHistogram(const unsigned char *h_image, int *h_hist, int size)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++)
    {
        h_hist[h_image[i]]++;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "CPU time: " << time << " ms" << std::endl;
}

int main()
{
    // img and hist size in cpu
    const int imgSize = SIZE;
    size_t imgBytes = imgSize * sizeof(unsigned char);
    size_t histBytes = NUM_BINS * sizeof(int);

    // img and hist in cpu
    unsigned char *h_image = new unsigned char[imgSize];
    int *h_hist = new int[NUM_BINS]();
    int *h_hist_gpu = new int[NUM_BINS]();

    // random pixel initialization
    for (int i = 0; i < imgSize; i++)
    {
        h_image[i] = rand() % NUM_BINS;
    }

    // mem allocation in gpu
    unsigned char *d_image;
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

    // kernel
    int threads = TILE;
    int wpt = 4;
    int blocks = ((imgSize + threads - 1) / threads) / wpt;

    cudaEventRecord(startKernel);
    kernel<<<blocks, threads>>>(d_image, d_hist, wpt);
    cudaEventRecord(stopKernel);
    // cudaDeviceSynchronize();

    //(D2H) timing
    cudaEventRecord(startD2H);
    cudaMemcpy(h_hist_gpu, d_hist, histBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);

    // cpu validation func call
    cpuHistogram(h_image, h_hist, imgSize);

    // assert values
    for (int i = 0; i < NUM_BINS; i++)
    {
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
