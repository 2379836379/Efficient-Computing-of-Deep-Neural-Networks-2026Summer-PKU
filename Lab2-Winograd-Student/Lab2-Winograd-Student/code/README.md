# Lab 02: Strassen & Winograd — Submission Guide

## Submission Format

- **File**: `Lab2_<StudentID>.zip`
- **Structure**:
  ```
  Lab2_<StudentID>/
    |- report.pdf                 (max 4 pages)
    |- code/
    |    |- strassen_impl.cpp (Q1)
    |    |- winograd_impl.cpp (Q2)
    |    |- Makefile
    |- README.txt                 (optional notes)
  ```

## Constraints

- **No external libraries** — only the C++ standard library (no Armadillo, Eigen, BLAS, etc.).
- **Do not modify** the required function signatures (listed below).
- You **may** add helper functions, structs, or additional optimizations.
- Compilation must succeed with `g++ -std=c++17 -O3 -Wall`.

## Q1: Strassen (`strassen_impl.cpp`)

**Required interfaces** (do not change signatures):

| Function | Purpose |
|---|---|
| `void matrix_add(float** A, float** B, float** C, int n)` | Element-wise addition of n×n matrices |
| `void matrix_sub(float** A, float** B, float** C, int n)` | Element-wise subtraction of n×n matrices |
| `void strassen(float** A, float** B, float** C, int n)` | Strassen multiply for n×n (n is power of 2) |
| `void strassen_multiply(const float* A, int I, int K, const float* B, int K2, int J, float* C)` | Top-level entry: handles padding for non-power-of-2 |

**How to use**:

1. Open `strassen_impl.cpp` and implement all `TODO` sections.
2. Build: `make strassen` (or `g++ -std=c++17 -O3 -Wall -o strassen strassen_impl.cpp`)
3. Run: `./strassen` — benchmarks n = 256, 512, 1024 and verifies correctness automatically.

## Q2: Winograd (`winograd_impl.cpp`)

**Required interfaces** (do not change signatures):

| Function | Purpose |
|---|---|
| `void kernel_transform(const float* g, float* U)` | Compute U = G · g · G^T (3×3 → 4×4) |
| `void input_transform(const float* d, int tile_stride, float* V)` | Compute V = B^T · d · B (4×4 tile → 4×4) |
| `void output_transform(const float* M, float* Y, int out_stride)` | Compute Y = A^T · M · A (4×4 → 2×2) |
| `void winograd_conv2d(const float* input, int C_in, int H, int W, const float* kernel, int C_out, int K_size, float* output, int stride, int pad)` | Full Winograd F(2×2, 3×3) convolution |

**How to use**:

1. Open `winograd_impl.cpp` and implement all `TODO` sections.
2. Build: `make winograd` (or `g++ -std=c++17 -O3 -Wall -o winograd winograd_impl.cpp`)
3. Run: `./winograd` — benchmarks against direct convolution and verifies correctness.

**Transform matrices** (B^T, G, A^T) are already defined as constants in the template.

## Report (`report.pdf`)

1. **Q1**: Runtime table (Naive vs. Strassen for n ∈ {256, 512, 1024}), speedup analysis.
2. **Q2**: Runtime table (Direct/im2col vs. Winograd), analysis of why it is (or isn't) faster.
3. **Correctness**: Max error vs. reference for both implementations.
