#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

const int N = 1 << 24; 
const int bytes = N * sizeof(float);

__global__ void doubleElements(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= 2.0f;
}

void checkResult(float* ref, float* gpu, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(ref[i] - gpu[i]) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": " << ref[i] << " vs " << gpu[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct \n";
}

int main() {
    // Allocate host memory
    float *h_src = new float[N];
    float *h_dst_sequential = new float[N];
    float *h_dst_streams = new float[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, bytes);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========================
    // 1. Sequential version (default stream)
    // ========================
    cudaMemset(d_data, 0, bytes);

    cudaEventRecord(start);

    cudaMemcpy(d_data, h_src, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    doubleElements<<<blocks, threads>>>(d_data, N);

    cudaMemcpy(h_dst_sequential, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_sequential = 0;
    cudaEventElapsedTime(&time_sequential, start, stop);

    std::cout << "Sequential time (default stream): " << time_sequential << " ms\n";

    // ========================
    // 2. Overlapping version (using streams)
    // ========================
    cudaMemset(d_data, 0, bytes);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int halfN = N / 2;
    int halfBytes = bytes / 2;

    cudaEventRecord(start);

    // First half in stream1
    cudaMemcpyAsync(d_data, h_src, halfBytes, cudaMemcpyHostToDevice, stream1);
    doubleElements<<<(halfN + threads - 1) / threads, threads, 0, stream1>>>(d_data, halfN);
    cudaMemcpyAsync(h_dst_streams, d_data, halfBytes, cudaMemcpyDeviceToHost, stream1);

    // Second half in stream2
    cudaMemcpyAsync(d_data + halfN, h_src + halfN, halfBytes, cudaMemcpyHostToDevice, stream2);
    doubleElements<<<(halfN + threads - 1) / threads, threads, 0, stream2>>>(d_data + halfN, halfN);
    cudaMemcpyAsync(h_dst_streams + halfN, d_data + halfN, halfBytes, cudaMemcpyDeviceToHost, stream2);

    // Wait for all to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_streams = 0;
    cudaEventElapsedTime(&time_streams, start, stop);

    std::cout << "Overlapping time (with streams): " << time_streams << " ms\n";

    // ========================
    // 3. Validate
    // ========================
    // Reference CPU computation
    for (int i = 0; i < N; ++i) {
        h_src[i] *= 2.0f;
    }

    std::cout << "Checking sequential result... ";
    checkResult(h_src, h_dst_sequential, N);

    std::cout << "Checking streams result... ";
    checkResult(h_src, h_dst_streams, N);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
    delete[] h_src;
    delete[] h_dst_sequential;
    delete[] h_dst_streams;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
