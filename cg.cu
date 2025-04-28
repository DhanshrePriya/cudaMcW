#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define N (1 << 22)
#define BLOCK_SIZE 256

__global__ void coop_reduce(int *data, int *result, int n) {
    cg::thread_block       tb = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

    extern __shared__ int warp_sums[];         

    int tid      = threadIdx.x;
    int idx      = blockIdx.x * blockDim.x + tid;
    int warps    = (blockDim.x + 31) / 32;      // # of warps this block
    int lane     = tile.thread_rank();         // 0–31 within warp
    int warp_id  = tid / 32;                   // which warp in block

    // load or zero
    int val = (idx < n) ? data[idx] : 0;

    // intra‑warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += tile.shfl_down(val, offset);
    }

    // first lane of each warp writes its sum
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    cg::sync(tb);

    // let the first warp reduce warp_sums[]
    if (warp_id == 0) {
        val = (lane < warps) ? warp_sums[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += tile.shfl_down(val, offset);
        }
        if (lane == 0) {
            result[blockIdx.x] = val;
        }
    }
}

// Recursively call coop_reduce until one value remains
int gpu_reduce_coop(int *d_in, int *d_tmp, int n, float &ms) {
    int *in = d_in, *out = d_tmp;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int warps = (BLOCK_SIZE + 31) / 32;
    size_t shmem = warps * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (blocks > 1) {
        coop_reduce<<<blocks, BLOCK_SIZE, shmem>>>(in, out, n);
        cudaDeviceSynchronize();
        n = blocks;
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::swap(in, out);
    }
    coop_reduce<<<1, BLOCK_SIZE, shmem>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    int result;
    cudaMemcpy(&result, out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

int main() {
    // prepare host data
    int *h = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h[i] = rand() % 10;

    // allocate device
    int *d_in, *d_tmp;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_tmp, N * sizeof(int));
    cudaMemcpy(d_in, h, N * sizeof(int), cudaMemcpyHostToDevice);

    // CPU sum
    long long cpu = 0;
    for (int i = 0; i < N; i++) cpu += h[i];

    // GPU coop reduction
    float time_ms = 0;
    int gpu = gpu_reduce_coop(d_in, d_tmp, N, time_ms);

    // report
    printf("CPU Sum:  %lld\n", cpu);
    printf("Coop Sum: %d  (%.3f ms)  Match: %s\n",
           gpu, time_ms, (gpu == cpu) ? "YES" : "NO");

    cudaFree(d_in);
    cudaFree(d_tmp);
    free(h);
    return 0;
}
