#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

void MatrixMulCPU(int m, int n, int k, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

int main() {
    int m = 16384, n = 16384, k = 16384; 

    vector<float> A(m * n, 1.0f);
    vector<float> B(n * k, 1.0f);
    vector<float> C(m * k, 0.0f);

    auto start = chrono::high_resolution_clock::now();
    MatrixMulCPU(m, n, k, A, B, C);
    auto end = chrono::high_resolution_clock::now();

    double duration = chrono::duration<double, milli>(end - start).count();
    cout << "CPU Time: " << duration << " ms" << endl;

    return 0;
}
