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
#include <algorithm>

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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// ============================================================
//  TODO: Matrix subtraction  C = A - B  (n x n)
// ============================================================
void matrix_sub(float** A, float** B, float** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// ============================================================
//  TODO: Strassen algorithm for n x n square matrices
//        C = A * B
//        n is guaranteed to be a power of 2.
//        Base case: when n <= threshold, fall back to naive multiply.
// ============================================================
void strassen(float** A, float** B, float** C, int n) {
    if (n <= 64) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return;
    }

    const int m = n / 2;

    auto copy_block = [m](float** src, int row0, int col0) {
        float** dst = alloc_matrix(m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                dst[i][j] = src[row0 + i][col0 + j];
            }
        }
        return dst;
    };

    float** A11 = copy_block(A, 0, 0);
    float** A12 = copy_block(A, 0, m);
    float** A21 = copy_block(A, m, 0);
    float** A22 = copy_block(A, m, m);
    float** B11 = copy_block(B, 0, 0);
    float** B12 = copy_block(B, 0, m);
    float** B21 = copy_block(B, m, 0);
    float** B22 = copy_block(B, m, m);

    float** T1 = alloc_matrix(m);
    float** T2 = alloc_matrix(m);
    float** P1 = alloc_matrix(m);
    float** P2 = alloc_matrix(m);
    float** P3 = alloc_matrix(m);
    float** P4 = alloc_matrix(m);
    float** P5 = alloc_matrix(m);
    float** P6 = alloc_matrix(m);
    float** P7 = alloc_matrix(m);

    matrix_add(A11, A22, T1, m);
    matrix_add(B11, B22, T2, m);
    strassen(T1, T2, P1, m);

    matrix_add(A21, A22, T1, m);
    strassen(T1, B11, P2, m);

    matrix_sub(B12, B22, T2, m);
    strassen(A11, T2, P3, m);

    matrix_sub(B21, B11, T2, m);
    strassen(A22, T2, P4, m);

    matrix_add(A11, A12, T1, m);
    strassen(T1, B22, P5, m);

    matrix_sub(A21, A11, T1, m);
    matrix_add(B11, B12, T2, m);
    strassen(T1, T2, P6, m);

    matrix_sub(A12, A22, T1, m);
    matrix_add(B21, B22, T2, m);
    strassen(T1, T2, P7, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            C[i][j] = P1[i][j] + P4[i][j] - P5[i][j] + P7[i][j];
            C[i][j + m] = P3[i][j] + P5[i][j];
            C[i + m][j] = P2[i][j] + P4[i][j];
            C[i + m][j + m] = P1[i][j] - P2[i][j] + P3[i][j] + P6[i][j];
        }
    }

    free_matrix(A11, m);
    free_matrix(A12, m);
    free_matrix(A21, m);
    free_matrix(A22, m);
    free_matrix(B11, m);
    free_matrix(B12, m);
    free_matrix(B21, m);
    free_matrix(B22, m);
    free_matrix(T1, m);
    free_matrix(T2, m);
    free_matrix(P1, m);
    free_matrix(P2, m);
    free_matrix(P3, m);
    free_matrix(P4, m);
    free_matrix(P5, m);
    free_matrix(P6, m);
    free_matrix(P7, m);
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

    int N = 1;
    int mx = std::max(I, std::max(K, J));
    while (N < mx) N <<= 1;

    float** Ap = alloc_matrix(N);
    float** Bp = alloc_matrix(N);
    float** Cp = alloc_matrix(N);

    for (int i = 0; i < I; i++) {
        for (int k = 0; k < K; k++) {
            Ap[i][k] = A[i * K + k];
        }
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < J; j++) {
            Bp[k][j] = B[k * J + j];
        }
    }

    strassen(Ap, Bp, Cp, N);

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            C[i * J + j] = Cp[i][j];
        }
    }

    free_matrix(Ap, N);
    free_matrix(Bp, N);
    free_matrix(Cp, N);
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
