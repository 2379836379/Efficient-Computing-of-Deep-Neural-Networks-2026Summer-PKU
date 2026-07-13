// CMSC5743 Lab 02 - Q1: Strassen Matrix Multiplication
// Template file: implement all functions marked with TODO.
//
// Compile: g++ strassen_impl.cpp -o strassen -std=c++17 -O3 -Wall
// Run:     ./strassen
//
// Requirements:
//   - Implement Strassen algorithm for matrix multiplication C = A * B
//   - A is I x K, B is K x J (I, K, J in {256, 512, 1024})
//   - Compare runtime with naive matmul() from Lab 01
//   - Report speedup for each matrix size

#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// ============================================================
//  Helper: allocate / free n x n float matrix
// ============================================================
float** alloc_matrix(int n) {
    float** M = new float*[n];
    for (int i = 0; i < n; i++) {
        M[i] = new float[n]();
    }
    return M;
}

void free_matrix(float** M, int n) {
    for (int i = 0; i < n; i++) delete[] M[i];
    delete[] M;
}

// ============================================================
//  TODO: Matrix addition  C = A + B  (n x n)
// ============================================================
void matrix_add(float** A, float** B, float** C, int n) {
    // TODO: implement
}

// ============================================================
//  TODO: Matrix subtraction  C = A - B  (n x n)
// ============================================================
void matrix_sub(float** A, float** B, float** C, int n) {
    // TODO: implement
}

// ============================================================
//  TODO: Strassen algorithm for n x n square matrices
//        C = A * B
//        n is guaranteed to be a power of 2.
//        Base case: when n <= threshold, fall back to naive multiply.
// ============================================================
void strassen(float** A, float** B, float** C, int n) {
    // Base case: use naive multiplication for small matrices
    // Recommended threshold: n <= 64

    // TODO: implement
    // 1. Split A and B into 4 sub-matrices each (A11, A12, A21, A22, ...)
    // 2. Compute 7 intermediate products S1..S7 using matrix_add/matrix_sub
    //    and recursive strassen() calls
    // 3. Combine S1..S7 into C11, C12, C21, C22
    // 4. Assemble C from its 4 sub-matrices
}

// ============================================================
//  TODO: Top-level Strassen multiply for non-square matrices
//        C (I x J) = A (I x K) * B (K x J)
//        Must handle non-power-of-2 dimensions by padding.
// ============================================================
void strassen_multiply(const float* A, int I, int K,
                       const float* B, int K2, int J,
                       float* C) {
    assert(K == K2);

    // TODO: implement
    // 1. Find N = next power of 2 >= max(I, K, J)
    // 2. Create N x N padded matrices for A, B, C (zero-padded)
    // 3. Call strassen() on the padded matrices
    // 4. Extract the I x J result into C
    // 5. Free temporary matrices
}

// ============================================================
//  Naive matmul for correctness verification
//  C (I x J) = A (I x K) * B (K x J)
//  A, B, C are stored in row-major order.
// ============================================================
void matmul_naive(const float* A, int I, int K,
                  const float* B, int K2, int J,
                  float* C) {
    assert(K == K2);
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * J + j];
            }
            C[i * J + j] = sum;
        }
    }
}

// ============================================================
//  Verify Strassen result against naive matmul
// ============================================================
void verify(const float* C_strassen, const float* C_naive, int I, int J) {
    float max_rel_err = 0.0f;
    for (int i = 0; i < I * J; i++) {
        float err = std::fabs(C_strassen[i] - C_naive[i]);
        float denom = std::max(std::fabs(C_naive[i]), 1.0f);
        float rel = err / denom;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    std::cout << "  Max relative error: " << max_rel_err;
    if (max_rel_err < 1e-3f)
        std::cout << "  [PASS]" << std::endl;
    else
        std::cout << "  [FAIL]" << std::endl;
}

// ============================================================
//  Benchmark
// ============================================================
int main() {
    const int sizes[] = {256, 512, 1024};
    const int num_runs = 5;

    srand(42);

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        int I = n, K = n, J = n;

        float* A = new float[I * K];
        float* B = new float[K * J];
        float* C_naive = new float[I * J];
        float* C_strassen = new float[I * J];

        for (int i = 0; i < I * K; i++) A[i] = (float)(rand() % 100) / 10.0f;
        for (int i = 0; i < K * J; i++) B[i] = (float)(rand() % 100) / 10.0f;

        // Benchmark naive
        double t_naive = 0;
        for (int r = 0; r < num_runs; r++) {
            double t = get_time();
            matmul_naive(A, I, K, B, K, J, C_naive);
            t_naive += get_time() - t;
        }
        t_naive /= num_runs;

        // Benchmark Strassen
        double t_strassen = 0;
        for (int r = 0; r < num_runs; r++) {
            double t = get_time();
            strassen_multiply(A, I, K, B, K, J, C_strassen);
            t_strassen += get_time() - t;
        }
        t_strassen /= num_runs;

        std::cout << "Matrix size: " << n << " x " << n << std::endl;
        std::cout << "  Naive:    " << t_naive    << " s" << std::endl;
        std::cout << "  Strassen: " << t_strassen << " s" << std::endl;
        std::cout << "  Speedup:  " << t_naive / t_strassen << "x" << std::endl;
        verify(C_strassen, C_naive, I, J);
        std::cout << std::endl;

        delete[] A;
        delete[] B;
        delete[] C_naive;
        delete[] C_strassen;
    }

    return 0;
}
