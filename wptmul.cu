// tiled_matmul_with_wpt_debug.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N            512     // Matrix dimension (N×N)
#define TILE_SIZE    32   // Tile width
#define WPT          2       // Work Per Thread (each thread computes WPT×WPT outputs)
#define EPSILON      1e-3f   // Tolerance for correctness check

// CUDA kernel: shared‐memory tiled matmul + WPT
__global__ void matmul_tiled_wpt(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global row/col of the first output element this thread will write
    int row = (blockIdx.y * TILE_SIZE + ty) * WPT;
    int col = (blockIdx.x * TILE_SIZE + tx) * WPT;

    // Accumulators for the WPT×WPT sub‐block
    float acc[WPT][WPT] = {0};

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load A tile into shared memory (each thread loads WPT elements)
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int globalRow = row + i;
            int globalCol = t * TILE_SIZE + tx;
            tile_A[ty * WPT + i][tx] =
                (globalRow < n && globalCol < n)
                    ? A[globalRow * n + globalCol]
                    : 0.0f;
        }
        // Load B tile into shared memory
        #pragma unroll
        for (int j = 0; j < WPT; ++j) {
            int globalRow = t * TILE_SIZE + ty;
            int globalCol = col + j;
            tile_B[ty][tx * WPT + j] =
                (globalRow < n && globalCol < n)
                    ? B[globalRow * n + globalCol]
                    : 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                #pragma unroll
                for (int j = 0; j < WPT; ++j) {
                    acc[i][j] +=
                        tile_A[ty * WPT + i][k] *
                        tile_B[k][tx * WPT + j];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < WPT; ++i) {
        for (int j = 0; j < WPT; ++j) {
            int r = row + i;
            int c = col + j;
            if (r < n && c < n) {
                C[r * n + c] = acc[i][j];
            }
        }
    }
}

// CPU‐side reference matmul
void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[r * n + k] * B[k * n + c];
            }
            C[r * n + c] = sum;
        }
    }
}

// Verify GPU result against CPU reference
bool verify(const float* gpu, const float* cpu, int n) {
    for (int i = 0; i < n * n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > EPSILON) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes),
          *h_B = (float*)malloc(bytes),
          *h_C = (float*)malloc(bytes),
          *h_ref = (float*)malloc(bytes);

    // Initialize inputs
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Create CUDA events
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    // H2D timing
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    // Launch parameters
    dim3 blockDim(TILE_SIZE / WPT, TILE_SIZE / WPT);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    // Kernel timing + launch + debug check
    cudaEventRecord(start_kernel);
    matmul_tiled_wpt<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_kernel);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    // Make sure kernel has finished before we do D2H timing
    cudaDeviceSynchronize();

    // D2H timing
    cudaEventRecord(start_d2h);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);

    // Wait for all events
    cudaEventSynchronize(stop_h2d);
    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop_d2h);

    // Compute elapsed times
    float tH2D, tKernel, tD2H;
    cudaEventElapsedTime(&tH2D,     start_h2d,   stop_h2d);
    cudaEventElapsedTime(&tKernel,  start_kernel, stop_kernel);
    cudaEventElapsedTime(&tD2H,     start_d2h,   stop_d2h);

    printf("H2D Copy Time  : %f ms\n", tH2D);
    printf("Kernel Time    : %f ms\n", tKernel);
    printf("D2H Copy Time  : %f ms\n", tD2H);


    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
