#include <iostream>
using namespace std;
#include <vector>
#include <cuda_runtime.h>

// kernel to transpose matrix 
__global__ void transposeKernel(float* input, float* output, int P, int Q) {
    const int TILE_X = 4;
    const int TILE_Y = 4;
    __shared__ float tile[TILE_Y][TILE_X]; 

    int x = blockIdx.x * TILE_X + threadIdx.x;
    int y = blockIdx.y * TILE_Y + threadIdx.y;

    // load matrix as such 
    if (x < P && y < Q) {
        tile[threadIdx.y][threadIdx.x] = input[y * P + x];
    }
    __syncthreads();

    // transpose to indoces to point to the correspondingly opposite block diagonally
    x = blockIdx.y * TILE_Y + threadIdx.x;
    y = blockIdx.x * TILE_X + threadIdx.y;

    // write into the output matrix in coalesced manner
    if (x < Q && y < P) {
        output[y * Q + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// cpu transpose 
void cpuTranspose(const std::vector<float>& input, std::vector<float>& output, int P, int Q) {
    for (int i = 0; i < Q; i++) {
        for (int j = 0; j < P; j++) {
            output[j * Q + i] = input[i * P + j]; // rowid * no of cols + colid
        }
    }
}

// initilize with random values
void initMatrix(vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand() % 100);
    }
}

// check correctness using fabs
bool verifyResults(const vector<float>& cpuRes, const vector<float>& gpuRes, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) { /// aboslute floating point check for 3 decimal places 
            cout << "Mismatch at index " << i << ": CPU=" << cpuRes[i] << ", GPU=" << gpuRes[i] << endl;
            return false;
        }
    }
    return true;
}

// print
void printMatrix(const vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) { 
        for (int j = 0; j < cols; j++) {
            cout << mat[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    // dimesion of the square matrix 
    int P = 1024;  // no of cols
    int Q = 87; // no of rows
    size_t size = P * Q * sizeof(float);

    // vector declaration
    vector<float> h_input(P * Q);
    vector<float> h_output(Q * P);
    vector<float> h_output_cpu(Q * P);

    // initialsed to random values 
    initMatrix(h_input, Q, P);

    // gpu mem allocation 
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // copy input vector to device 
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    //kernel lauch 
    dim3 blockDim(4, 4);
    dim3 gridDim((P + 3) / 4, (Q + 3) / 4);
    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, P, Q);

    // copy op from gpu 
    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    // verify results 
    cpuTranspose(h_input, h_output_cpu, P, Q);

    if (verifyResults(h_output_cpu, h_output, P * Q)) {
        cout << "CORRECT!" << endl;
    } else {
        cout << "INCORRECT!" << endl;
    }

    // print
    // cout << "Original Matrix:" << endl;
    // printMatrix(h_input, Q, P);
    // cout << "GPU Transposed Matrix:" << endl;
    // printMatrix(h_output, P, Q);
    // cout << "CPU Transposed Matrix:" << endl;
    // printMatrix(h_output_cpu, P, Q);

    // free up mem 
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}