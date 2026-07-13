// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <immintrin.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

struct GemmProblem {
  int I;
  int J;
  int K;
};

using Matrix = std::vector<int>;

static inline int idx(int row, int col, int stride) {
  return row * stride + col;
}

void fill_random(Matrix &m) {
  for (auto &x : m) {
    x = rand() % 13 - 6;
  }
}

void zero(Matrix &m) {
  std::fill(m.begin(), m.end(), 0);
}

void gemm_groundtruth(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < p.J; ++j) {
      int sum = 0;
      for (int k = 0; k < p.K; ++k) {
        sum += A[idx(i, k, p.K)] * B[idx(k, j, p.J)];
      }
      C[idx(i, j, p.J)] = sum;
    }
  }
}

void gemm_ijk(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < p.J; ++j) {
      for (int k = 0; k < p.K; ++k) {
        C[idx(i, j, p.J)] += A[idx(i, k, p.K)] * B[idx(k, j, p.J)];
      }
    }
  }
}

void gemm_ikj(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  for (int i = 0; i < p.I; ++i) {
    for (int k = 0; k < p.K; ++k) {
      const int a = A[idx(i, k, p.K)];
      for (int j = 0; j < p.J; ++j) {
        C[idx(i, j, p.J)] += a * B[idx(k, j, p.J)];
      }
    }
  }
}

void transpose_A_then_gemm(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  Matrix AT(p.I * p.K);
  zero(C);
  for (int i = 0; i < p.I; ++i) {
    for (int k = 0; k < p.K; ++k) {
      AT[idx(k, i, p.I)] = A[idx(i, k, p.K)];
    }
  }
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < p.J; ++j) {
      for (int k = 0; k < p.K; ++k) {
        C[idx(i, j, p.J)] += AT[idx(k, i, p.I)] * B[idx(k, j, p.J)];
      }
    }
  }
}

void transpose_B_then_gemm(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  Matrix BT(p.K * p.J);
  zero(C);
  for (int k = 0; k < p.K; ++k) {
    for (int j = 0; j < p.J; ++j) {
      BT[idx(j, k, p.K)] = B[idx(k, j, p.J)];
    }
  }
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < p.J; ++j) {
      for (int k = 0; k < p.K; ++k) {
        C[idx(i, j, p.J)] += A[idx(i, k, p.K)] * BT[idx(j, k, p.K)];
      }
    }
  }
}

void gemm_ikj_tiled(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p, int tile) {
  zero(C);
  for (int ii = 0; ii < p.I; ii += tile) {
    for (int kk = 0; kk < p.K; kk += tile) {
      for (int jj = 0; jj < p.J; jj += tile) {
        const int i_end = std::min(ii + tile, p.I);
        const int k_end = std::min(kk + tile, p.K);
        const int j_end = std::min(jj + tile, p.J);
        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            const int a = A[idx(i, k, p.K)];
            for (int j = jj; j < j_end; ++j) {
              C[idx(i, j, p.J)] += a * B[idx(k, j, p.J)];
            }
          }
        }
      }
    }
  }
}

void gemm_ikj_unrolled4(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  const int j4 = p.J & ~3;
  for (int i = 0; i < p.I; ++i) {
    for (int k = 0; k < p.K; ++k) {
      const int a = A[idx(i, k, p.K)];
      int j = 0;
      for (; j < j4; j += 4) {
        C[idx(i, j + 0, p.J)] += a * B[idx(k, j + 0, p.J)];
        C[idx(i, j + 1, p.J)] += a * B[idx(k, j + 1, p.J)];
        C[idx(i, j + 2, p.J)] += a * B[idx(k, j + 2, p.J)];
        C[idx(i, j + 3, p.J)] += a * B[idx(k, j + 3, p.J)];
      }
      for (; j < p.J; ++j) {
        C[idx(i, j, p.J)] += a * B[idx(k, j, p.J)];
      }
    }
  }
}

void gemm_write_cached(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < p.J; ++j) {
      int sum = 0;
      for (int k = 0; k < p.K; ++k) {
        sum += A[idx(i, k, p.K)] * B[idx(k, j, p.J)];
      }
      C[idx(i, j, p.J)] = sum;
    }
  }
}

void gemm_packed_B(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  Matrix packedB(p.K * p.J);
  zero(C);
  for (int k = 0; k < p.K; ++k) {
    for (int j = 0; j < p.J; ++j) {
      packedB[idx(k, j, p.J)] = B[idx(k, j, p.J)];
    }
  }
  for (int i = 0; i < p.I; ++i) {
    for (int k = 0; k < p.K; ++k) {
      const int a = A[idx(i, k, p.K)];
      const int *bp = &packedB[idx(k, 0, p.J)];
      for (int j = 0; j < p.J; ++j) {
        C[idx(i, j, p.J)] += a * bp[j];
      }
    }
  }
}

void gemm_simd4(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  zero(C);
  const int j4 = p.J & ~3;
  for (int i = 0; i < p.I; ++i) {
    for (int j = 0; j < j4; j += 4) {
      int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      #pragma GCC ivdep
      for (int k = 0; k < p.K; ++k) {
        const int a = A[idx(i, k, p.K)];
        s0 += a * B[idx(k, j + 0, p.J)];
        s1 += a * B[idx(k, j + 1, p.J)];
        s2 += a * B[idx(k, j + 2, p.J)];
        s3 += a * B[idx(k, j + 3, p.J)];
      }
      C[idx(i, j + 0, p.J)] = s0;
      C[idx(i, j + 1, p.J)] = s1;
      C[idx(i, j + 2, p.J)] = s2;
      C[idx(i, j + 3, p.J)] = s3;
    }
    for (int j = j4; j < p.J; ++j) {
      int sum = 0;
      for (int k = 0; k < p.K; ++k) {
        sum += A[idx(i, k, p.K)] * B[idx(k, j, p.J)];
      }
      C[idx(i, j, p.J)] = sum;
    }
  }
}

void gemm_hybrid_fast(const Matrix &A, const Matrix &B, Matrix &C, const GemmProblem &p) {
  using v4i = int __attribute__((vector_size(16)));
  zero(C);
  constexpr int tile = 32;
  for (int ii = 0; ii < p.I; ii += tile) {
    const int i_end = std::min(ii + tile, p.I);
    for (int kk = 0; kk < p.K; kk += tile) {
      const int k_end = std::min(kk + tile, p.K);
      for (int jj = 0; jj < p.J; jj += tile) {
        const int j_end = std::min(jj + tile, p.J);
        const int k_blk = k_end - kk;
        const int j_blk = j_end - jj;
        Matrix Bblk(k_blk * j_blk);
        for (int k = 0; k < k_blk; ++k) {
          for (int j = 0; j < j_blk; ++j) {
            Bblk[k * j_blk + j] = B[idx(kk + k, jj + j, p.J)];
          }
        }
        const int j4 = j_blk & ~3;
        for (int i = ii; i < i_end; ++i) {
          int *cptr = &C[idx(i, jj, p.J)];
          for (int k = 0; k < k_blk; ++k) {
            const int a = A[idx(i, kk + k, p.K)];
            const int *bptr = &Bblk[k * j_blk];
            v4i avec = {a, a, a, a};
            int j = 0;
            for (; j < j4; j += 4) {
              v4i cvec = *reinterpret_cast<v4i *>(cptr + j);
              v4i bvec = *reinterpret_cast<const v4i *>(bptr + j);
              cvec += avec * bvec;
              *reinterpret_cast<v4i *>(cptr + j) = cvec;
            }
            for (; j < j_blk; ++j) {
              cptr[j] += a * bptr[j];
            }
          }
        }
      }
    }
  }
}

struct Conv2DConfig {
  int batch;
  int height;
  int width;
  int in_channels;
  int out_channels;
  int kernel;
  int stride;
  int padding;
};

static inline int conv_in_idx(int b, int c, int h, int w, const Conv2DConfig &cfg) {
  return ((b * cfg.in_channels + c) * cfg.height + h) * cfg.width + w;
}

static inline int conv_w_idx(int oc, int ic, int kh, int kw, const Conv2DConfig &cfg) {
  return (((oc * cfg.in_channels + ic) * cfg.kernel + kh) * cfg.kernel + kw);
}

static inline int conv_out_idx(int b, int oc, int h, int w, int out_h, int out_w, int out_channels) {
  return ((b * out_channels + oc) * out_h + h) * out_w + w;
}

void im2col(const std::vector<int> &input, std::vector<int> &col, const Conv2DConfig &cfg, int out_h, int out_w) {
  const int channels_col = cfg.in_channels * cfg.kernel * cfg.kernel;
  for (int b = 0; b < cfg.batch; ++b) {
    for (int c = 0; c < cfg.in_channels; ++c) {
      for (int kh = 0; kh < cfg.kernel; ++kh) {
        for (int kw = 0; kw < cfg.kernel; ++kw) {
          const int col_c = (c * cfg.kernel + kh) * cfg.kernel + kw;
          for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
              const int ih = oh * cfg.stride + kh - cfg.padding;
              const int iw = ow * cfg.stride + kw - cfg.padding;
              int value = 0;
              if (0 <= ih && ih < cfg.height && 0 <= iw && iw < cfg.width) {
                value = input[conv_in_idx(b, c, ih, iw, cfg)];
              }
              const int col_row = b * channels_col * out_h * out_w + col_c * out_h * out_w + oh * out_w + ow;
              col[col_row] = value;
            }
          }
        }
      }
    }
  }
}

void conv2d_im2col(const std::vector<int> &input, const std::vector<int> &weight, std::vector<int> &output, const Conv2DConfig &cfg) {
  const int out_h = (cfg.height + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;
  const int out_w = (cfg.width + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;
  const int k = cfg.in_channels * cfg.kernel * cfg.kernel;
  const int n = out_h * out_w;
  std::vector<int> col(cfg.batch * k * n);
  im2col(input, col, cfg, out_h, out_w);

  output.assign(cfg.batch * cfg.out_channels * out_h * out_w, 0);
  for (int b = 0; b < cfg.batch; ++b) {
    for (int oc = 0; oc < cfg.out_channels; ++oc) {
      for (int p = 0; p < n; ++p) {
        int sum = 0;
        for (int kk = 0; kk < k; ++kk) {
          sum += weight[oc * k + kk] * col[b * k * n + kk * n + p];
        }
        output[conv_out_idx(b, oc, p / out_w, p % out_w, out_h, out_w, cfg.out_channels)] = sum;
      }
    }
  }
}

void conv2d_direct(const std::vector<int> &input, const std::vector<int> &weight, std::vector<int> &output, const Conv2DConfig &cfg) {
  const int out_h = (cfg.height + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;
  const int out_w = (cfg.width + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;
  output.assign(cfg.batch * cfg.out_channels * out_h * out_w, 0);
  for (int b = 0; b < cfg.batch; ++b) {
    for (int oc = 0; oc < cfg.out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int sum = 0;
          for (int ic = 0; ic < cfg.in_channels; ++ic) {
            for (int kh = 0; kh < cfg.kernel; ++kh) {
              for (int kw = 0; kw < cfg.kernel; ++kw) {
                const int ih = oh * cfg.stride + kh - cfg.padding;
                const int iw = ow * cfg.stride + kw - cfg.padding;
                if (0 <= ih && ih < cfg.height && 0 <= iw && iw < cfg.width) {
                  sum += input[conv_in_idx(b, ic, ih, iw, cfg)] * weight[conv_w_idx(oc, ic, kh, kw, cfg)];
                }
              }
            }
          }
          output[conv_out_idx(b, oc, oh, ow, out_h, out_w, cfg.out_channels)] = sum;
        }
      }
    }
  }
}

bool same_matrix(const Matrix &a, const Matrix &b) {
  return a == b;
}

using GemmFn = void (*)(const Matrix &, const Matrix &, Matrix &, const GemmProblem &);

void run_q1(int I, int J, int K, const std::string &algo, int repeat) {
  GemmProblem p{I, J, K};
  Matrix A(I * K), B(K * J), C(I * J), ref(I * J);
  fill_random(A);
  fill_random(B);
  gemm_groundtruth(A, B, ref, p);

  const struct {
    const char *name;
    GemmFn fn;
  } cases[] = {
      {"matmul_ijk", gemm_ijk},
      {"matmul_ikj", gemm_ikj},
      {"matmul_AT", transpose_A_then_gemm},
      {"matmul_BT", transpose_B_then_gemm},
      {"writing_caching", gemm_write_cached},
      {"tiling", [](const Matrix &a, const Matrix &b, Matrix &c, const GemmProblem &p) { gemm_ikj_tiled(a, b, c, p, 32); }},
      {"loop_unrolling", gemm_ikj_unrolled4},
      {"array_packing", gemm_packed_B},
      {"vectorization", gemm_simd4},
      {"hybrid_fast", gemm_hybrid_fast},
  };

  for (const auto &c : cases) {
    if (algo != "all" && algo != c.name) {
      continue;
    }
    double total = 0.0;
    for (int r = 0; r < repeat; ++r) {
      double t = get_time();
      c.fn(A, B, C, p);
      total += get_time() - t;
    }
    assert(same_matrix(C, ref));
    std::printf("%s I=%d J=%d K=%d avg_time=%.6f\n", c.name, I, J, K, total / repeat);
  }
}

void run_q2() {
  Conv2DConfig cfg{1, 56, 56, 3, 64, 3, 1, 0};
  const int out_h = (cfg.height + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;
  const int out_w = (cfg.width + 2 * cfg.padding - cfg.kernel) / cfg.stride + 1;

  std::vector<int> input(cfg.batch * cfg.in_channels * cfg.height * cfg.width);
  std::vector<int> weight(cfg.out_channels * cfg.in_channels * cfg.kernel * cfg.kernel);
  std::vector<int> out_direct;
  std::vector<int> out_im2col;

  fill_random(input);
  fill_random(weight);

  double t = get_time();
  conv2d_direct(input, weight, out_direct, cfg);
  double direct_time = get_time() - t;

  t = get_time();
  conv2d_im2col(input, weight, out_im2col, cfg);
  double im2col_time = get_time() - t;

  assert(out_direct == out_im2col);
  std::printf("conv_direct avg_time=%.6f\n", direct_time);
  std::printf("conv_im2col avg_time=%.6f\n", im2col_time);
  std::printf("output_shape=%d x %d x %d x %d\n", cfg.batch, cfg.out_channels, out_h, out_w);
}

void run_q3() {
  GemmProblem p{1024, 1024, 1024};
  Matrix A(p.I * p.K), B(p.K * p.J), C(p.I * p.J), ref(p.I * p.J);
  fill_random(A);
  fill_random(B);
  gemm_groundtruth(A, B, ref, p);

  const struct {
    const char *name;
    GemmFn fn;
  } cases[] = {
      {"loop_unrolling", gemm_ikj_unrolled4},
      {"writing_caching", gemm_write_cached},
      {"tiling", [](const Matrix &a, const Matrix &b, Matrix &c, const GemmProblem &p) { gemm_ikj_tiled(a, b, c, p, 32); }},
      {"vectorization", gemm_simd4},
      {"hybrid_fast", gemm_hybrid_fast},
      {"array_packing", gemm_packed_B},
  };

  for (const auto &c : cases) {
    double total = 0.0;
    for (int r = 0; r < 3; ++r) {
      double t = get_time();
      c.fn(A, B, C, p);
      total += get_time() - t;
    }
    assert(same_matrix(C, ref));
    std::printf("%s avg_time=%.6f\n", c.name, total / 3.0);
  }
}

int main(int argc, char **argv) {
  srand(0);
  std::string mode = argc >= 2 ? argv[1] : "q1";
  if (mode == "q1") {
    int I = argc >= 3 ? std::atoi(argv[2]) : 256;
    int J = argc >= 4 ? std::atoi(argv[3]) : I;
    int K = argc >= 5 ? std::atoi(argv[4]) : I;
    std::string algo = argc >= 6 ? argv[5] : "all";
    int repeat = argc >= 7 ? std::atoi(argv[6]) : 3;
    run_q1(I, J, K, algo, repeat);
  } else if (mode == "q2") {
    run_q2();
  } else if (mode == "q3") {
    run_q3();
  } else {
    std::fprintf(stderr, "Usage: %s [q1|q2|q3] ...\n", argv[0]);
    return 1;
  }
  return 0;
}





