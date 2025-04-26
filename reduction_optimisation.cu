#include <stdio.h>
#include <cuda.h>

#define N (1 << 26)  
#define BLOCK_SIZE 128

// Kernel 1: Interleaved Addressing (Divergent Branching)
__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 2: Interleaved Addressing (Avoiding Divergence)
__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x)
            sdata[index] += sdata[index + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 3: Sequential Addressing
__global__ void reduce3(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 4: First Add During Load
__global__ void reduce4(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 5: Unroll Last Warp
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 6: Completely Unrolled Reduction
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    sdata[tid] = g_idata[i] + g_idata[i + blockSize];
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 7: Multiple Elements Per Thread
__global__ void reduce7(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockDim.x];
        i += gridSize;
    }
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Benchmark function
void runBenchmark(void (*kernel)(int*, int*), int *d_idata, int *d_odata, int numBlocks, const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%s: %.3f ms\n", name, milliseconds);
}

// Special case for Kernel 7
void runBenchmark7(void (*kernel)(int*, int*, unsigned int), int *d_idata, int *d_odata, int numBlocks, unsigned int n, const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%s: %.3f ms\n", name, milliseconds);
}

int main() {
    int *h_idata = new int[N];
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, (N / BLOCK_SIZE) * sizeof(int));

    for (int i = 0; i < N; i++) h_idata[i] = 1;
    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Running Reduction Kernels...\n");
    runBenchmark(reduce1, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 1");
    runBenchmark(reduce2, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 2");
    runBenchmark(reduce3, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 3");
    runBenchmark(reduce4, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 4");
    runBenchmark(reduce5, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 5");
    runBenchmark(reduce6<128>, d_idata, d_odata, N / (BLOCK_SIZE * 2), "Kernel 6");
    runBenchmark7(reduce7, d_idata, d_odata, N / (BLOCK_SIZE * 2), N, "Kernel 7");

    cudaFree(d_idata);
    cudaFree(d_odata);
    delete[] h_idata;
}
