// CMSC5743 Lab 02 - Q2: Winograd Convolution (F(2x2, 3x3))
// Template file: implement all functions marked with TODO.
//
// Compile: g++ winograd_impl.cpp -o winograd -std=c++17 -O3 -Wall
// Run:     ./winograd
//
// Requirements:
//   - Implement Winograd F(2x2, 3x3) convolution from scratch in pure C++
//   - Do NOT use external linear algebra libraries (e.g., Armadillo, Eigen)
//   - Compare runtime with your im2col implementation from Lab 01
//   - Provide analysis on speed improvement (or lack thereof) and why
//
// Convolution parameters (same as Lab 01):
//   batch=1, H=56, W=56, C_in=3, C_out=64, kernel=3x3, stride=1, padding=0

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
//  Fixed Winograd F(2x2, 3x3) transform matrices
//
//  B^T (4x4): input transform
//  G   (4x3): kernel transform
//  A^T (2x4): output transform
//
//  Y = A^T [ (G g G^T) odot (B^T d B) ] A
// ============================================================

// B^T: input transform matrix (4x4)
const float BT[4][4] = {
    { 1,  0, -1,  0},
    { 0,  1,  1,  0},
    { 0, -1,  1,  0},
    { 0,  1,  0, -1}
};

// G: kernel transform matrix (4x3)
const float G[4][3] = {
    { 1.0f,    0.0f,   0.0f},
    { 0.5f,    0.5f,   0.5f},
    { 0.5f,   -0.5f,   0.5f},
    { 0.0f,    0.0f,   1.0f}
};

// A^T: output transform matrix (2x4)
const float AT[2][4] = {
    {1, 1,  1, 0},
    {0, 1, -1, -1}
};

// ============================================================
//  TODO: Transform a single 3x3 kernel into Winograd domain
//        U (4x4) = G * g * G^T
//        g: 3x3 kernel (row-major, stride = 3)
//        U: 4x4 output (row-major, stride = 4)
// ============================================================
void kernel_transform(const float* g, float* U) {
    float tmp[4][3];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += G[i][k] * g[k * 3 + j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += tmp[i][k] * G[j][k];
            }
            U[i * 4 + j] = sum;
        }
    }
}

// ============================================================
//  TODO: Transform a single 4x4 input tile into Winograd domain
//        V (4x4) = B^T * d * B
//        d: 4x4 input tile (row-major, stride = tile_stride)
//        V: 4x4 output (row-major, stride = 4)
// ============================================================
void input_transform(const float* d, int tile_stride, float* V) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += BT[i][k] * d[k * tile_stride + j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += tmp[i][k] * BT[j][k];
            }
            V[i * 4 + j] = sum;
        }
    }
}

// ============================================================
//  TODO: Inverse-transform Winograd domain result to spatial domain
//        Y (2x2) = A^T * M * A
//        M: 4x4 element-wise product (row-major)
//        Y: 2x2 output (row-major, stride = out_stride)
// ============================================================
void output_transform(const float* M, float* Y, int out_stride) {
    float tmp[2][4];
    const float A[4][2] = {
        {1, 0},
        {1, 1},
        {1, -1},
        {0, -1}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += AT[i][k] * M[k * 4 + j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += tmp[i][k] * A[k][j];
            }
            Y[i * out_stride + j] = sum;
        }
    }
}

// ============================================================
//  TODO: Winograd F(2x2, 3x3) convolution
//
//  input:   [C_in x H x W]       (row-major)
//  kernel:  [C_out x C_in x 3 x 3]  (row-major)
//  output:  [C_out x H_out x W_out]  (row-major)
//
//  H_out = H - 2,  W_out = W - 2  (stride=1, padding=0, kernel=3)
// ============================================================
void winograd_conv2d(const float* input,   int C_in,  int H,     int W,
                     const float* kernel,  int C_out, int K_size,
                     float* output,        int stride, int pad) {
    assert(K_size == 3 && stride == 1 && pad == 0);

    int H_out = H - 2;
    int W_out = W - 2;
    int num_tile_h = (H_out + 1) / 2;  // ceil(H_out / 2)
    int num_tile_w = (W_out + 1) / 2;  // ceil(W_out / 2)

    const int tile_size = 4;

    float* U = new float[C_out * C_in * tile_size * tile_size];
    float V[16];
    float M[16];

    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            const float* g = kernel + co * C_in * 9 + ci * 9;
            float* u = U + (co * C_in + ci) * 16;
            kernel_transform(g, u);
        }
    }

    std::memset(output, 0, sizeof(float) * C_out * H_out * W_out);

    for (int co = 0; co < C_out; co++) {
        for (int th = 0; th < num_tile_h; th++) {
            for (int tw = 0; tw < num_tile_w; tw++) {
                std::memset(M, 0, sizeof(M));
                int ih0 = th * 2;
                int iw0 = tw * 2;

                for (int ci = 0; ci < C_in; ci++) {
                    float tile[16];
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            int ih = ih0 + i;
                            int iw = iw0 + j;
                            tile[i * 4 + j] = (ih < H && iw < W)
                                ? input[ci * H * W + ih * W + iw]
                                : 0.0f;
                        }
                    }

                    input_transform(tile, 4, V);
                    const float* u = U + (co * C_in + ci) * 16;
                    for (int i = 0; i < 16; i++) {
                        M[i] += u[i] * V[i];
                    }
                }

                float out_tile_buf[4];
                output_transform(M, out_tile_buf, 2);

                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        int oh = ih0 + i;
                        int ow = iw0 + j;
                        if (oh < H_out && ow < W_out) {
                            output[co * H_out * W_out + oh * W_out + ow] = out_tile_buf[i * 2 + j];
                        }
                    }
                }
            }
        }
    }

    delete[] U;
}

// ============================================================
//  Direct convolution for correctness verification
// ============================================================
void conv2d_direct(const float* input,   int C_in,  int H,     int W,
                   const float* kernel,  int C_out, int K_size,
                   float* output,        int stride, int pad) {
    int H_out = (H + 2 * pad - K_size) / stride + 1;
    int W_out = (W + 2 * pad - K_size) / stride + 1;

    memset(output, 0, sizeof(float) * C_out * H_out * W_out);

    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < K_size; kh++) {
                        for (int kw = 0; kw < K_size; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input[ci * H * W + ih * W + iw]
                                     * kernel[co * C_in * K_size * K_size
                                              + ci * K_size * K_size
                                              + kh * K_size + kw];
                            }
                        }
                    }
                    output[co * H_out * W_out + oh * W_out + ow] += sum;
                }
            }
        }
    }
}

// ============================================================
//  Verification
// ============================================================
void verify(const float* out_wino, const float* out_ref, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = std::fabs(out_wino[i] - out_ref[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Max error: " << max_err;
    if (max_err < 1e-1f)
        std::cout << "  [PASS]" << std::endl;
    else
        std::cout << "  [FAIL]" << std::endl;
}

// ============================================================
//  Benchmark
// ============================================================
int main() {
    const int batch    = 1;
    const int H        = 56;
    const int W        = 56;
    const int C_in     = 3;
    const int C_out    = 64;
    const int K_size   = 3;
    const int stride   = 1;
    const int pad      = 0;
    const int H_out    = H - K_size + 1;
    const int W_out    = W - K_size + 1;
    const int num_runs = 10;
    (void)batch;

    srand(42);

    int input_size  = C_in * H * W;
    int kernel_size = C_out * C_in * K_size * K_size;
    int output_size = C_out * H_out * W_out;

    float* input   = new float[input_size];
    float* kernel  = new float[kernel_size];
    float* out_ref = new float[output_size];
    float* out_wino = new float[output_size];

    for (int i = 0; i < input_size;  i++) input[i]  = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < kernel_size; i++) kernel[i] = (float)(rand() % 100) / 50.0f;

    // Benchmark direct convolution
    double t_direct = 0;
    for (int r = 0; r < num_runs; r++) {
        double t = get_time();
        conv2d_direct(input, C_in, H, W, kernel, C_out, K_size, out_ref, stride, pad);
        t_direct += get_time() - t;
    }
    t_direct /= num_runs;

    // Benchmark Winograd convolution
    double t_wino = 0;
    for (int r = 0; r < num_runs; r++) {
        double t = get_time();
        winograd_conv2d(input, C_in, H, W, kernel, C_out, K_size, out_wino, stride, pad);
        t_wino += get_time() - t;
    }
    t_wino /= num_runs;

    std::cout << "Convolution: C_in=" << C_in << " C_out=" << C_out
              << " H=" << H << " W=" << W << " K=" << K_size << std::endl;
    std::cout << "  Direct:   " << t_direct << " s" << std::endl;
    std::cout << "  Winograd: " << t_wino   << " s" << std::endl;
    std::cout << "  Speedup:  " << t_direct / t_wino << "x" << std::endl;
    verify(out_wino, out_ref, output_size);

    delete[] input;
    delete[] kernel;
    delete[] out_ref;
    delete[] out_wino;

    return 0;
}
