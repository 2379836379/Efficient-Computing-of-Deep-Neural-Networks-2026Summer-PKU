# Lab 2 report

## 1.Q1: Strassen

测量朴素矩阵乘法和 Strassen 的平均运行时间。实际结果如下：

| Matrix Size | Naive Time (s) | Strassen Time (s) | Speedup |
|---|---:|---:|---:|
| 256 x 256 | 0.024407 | 0.00938497 | 2.60065x |
| 512 x 512 | 0.244691 | 0.063119 | 3.87666x |
| 1024 x 1024 | 5.18433 | 0.435526 | 11.9036x |

从表中可以看到，Strassen 在三个测试规模上都比朴素矩阵乘法更快，而且随着矩阵规模增大，速度优势逐渐扩大，原因主要有以下几点；

1. 乘法次数减少的收益在大规模时更明显：Strassen 每一层递归都把 8 次子乘法减少为 7 次。这一差异在规模小的时候并不显著，但随着递归深度增加，累计节省的乘法数量迅速放大，因此在 `1024 x 1024` 这类大矩阵上收益最明显。

2. **递归开销在小规模问题中占比更高**。  
Strassen 并不是“白赚”一次乘法，它需要额外的矩阵加减、子块复制、递归调用以及临时空间分配。因此，在小规模矩阵下，这些开销会抵消部分理论收益。当前结果中虽然 `256 x 256` 已经快于朴素算法，但加速比明显低于 `1024 x 1024`，这正体现了递归常数开销的影响。

3. **base case 的设置对性能影响很大**。  
本实现当 `n <= 64` 时使用朴素乘法，而不是继续递归。这样做减少了过细粒度递归产生的函数调用和内存管理成本。若阈值设置过小，递归层数增加，常数开销会更高；若阈值设置过大，又可能提前丢失 Strassen 的乘法优势。因此这个阈值本质上决定了实际中的 crossover point。

4. **cache 行为也会影响真实性能**。  
Strassen 把大矩阵拆成子块后，子问题规模逐步缩小，一定程度上更容易让局部数据留在 cache 中。但当前实现采用了较直接的子块复制与重新分配方式，并没有针对 cache locality 做特别优化，因此这里的 cache 收益并不是纯增益，还伴随额外的内存流量。即便如此，实验结果依然显示在较大规模下，减少乘法次数的收益足以压过这些开销。

总体来看，当前实现说明：**Strassen 在实践中可以更快，但这种更快依赖于足够大的问题规模和合适的递归阈值**。如果规模继续缩小，或者实现方式引入更多临时内存开销，那么它未必始终优于朴素乘法。


## Q2: Winograd

`code/winograd_impl.cpp` 中实现了以下接口：

- `kernel_transform(const float* g, float* U)`：计算 `U = G g G^T`。
- `input_transform(const float* d, int tile_stride, float* V)`：计算 `V = B^T d B`。
- `output_transform(const float* M, float* Y, int out_stride)`：计算 `Y = A^T M A`。
- `winograd_conv2d(...)`：完成完整的 Winograd 卷积流程。

Winograd `F(2x2, 3x3)` 的核心公式为：

`Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A`

其中：

- `g` 是 `3x3` 卷积核；
- `d` 是 `4x4` 输入 tile；
- `G g G^T` 是核变换；
- `B^T d B` 是输入变换；
- `⊙` 是逐元素乘法；
- `A^T (·) A` 是输出逆变换。

对一个 `2x2` 输出块，直接卷积通常需要 `2 x 2 x 3 x 3 = 36` 次标量乘法；而 Winograd 域中只需要 16 次逐元素乘法。因此，从 FLOPs 角度看，Winograd 的乘法数量更少，理论上更有优势。

本实现的卷积流程如下：

1. 先对所有卷积核做一次预变换，得到 Winograd 域中的 `U`。
2. 将输入按 `4x4` tile 划分，每个 tile 对应 `2x2` 输出块。
3. 对每个 tile 和每个输入通道做输入变换，得到 `V`。
4. 在 Winograd 域中做逐元素乘加，得到 `M`。
5. 对 `M` 做输出逆变换，恢复空间域 `2x2` 输出。

这个实现重点验证了算法流程正确性，但没有进一步做 SIMD、并行化、cache blocking 或更激进的内存布局优化，因此更适合作为算法实验版本，而不是最终高性能版本。


## 5. Q2 Analysis: Direct Convolution vs. Winograd

### 5.1 Runtime Comparison Table

Winograd 测试参数为：

- `C_in = 3`
- `C_out = 64`
- `H = 56`
- `W = 56`
- `K = 3`
- `stride = 1`
- `pad = 0`

实际测得运行结果如下：

| Method | Time (s) | Relative Speed |
|---|---:|---:|
| Direct Convolution | 0.000538278 | 1.00x |
| Winograd F(2x2, 3x3) | 0.0028877 | 0.186403x |

这里 `Relative Speed` 以直接卷积为基准，因此当前实现里 Winograd 并没有加速，反而更慢。

### 5.2 Performance Analysis

Winograd 理论上减少了乘法次数，但当前测试结果显示其性能明显落后于直接卷积。这说明减少 FLOPs 并不等价于减少运行时间，原因主要体现在以下几个方面。

第一，**FLOPs reduction 只减少了乘法，不减少全部成本**。  
Winograd 的优势主要来自将每个 `2x2` 输出块所需的乘法数量从 36 次降低到 16 次。但实际程序运行时间不仅由乘法决定，还包含输入变换、核变换、输出逆变换、循环控制和内存读写。对于一个未经深度优化的 CPU 实现，这些额外成本完全可能盖过理论上的 FLOPs 优势。

第二，**transform overhead 很明显**。  
在当前实现中：

- 每个卷积核都需要先做 `G g G^T` 变换；
- 每个输入 tile 都要做一次 `B^T d B`；
- 每个输出 tile 还要做 `A^T M A` 逆变换。

这些变换虽然主要是加减法，但都涉及多层小矩阵运算与局部数组操作，常数开销不低。对本实验这样规模不大的输入而言，transform overhead 占比很高，这是 Winograd 变慢的直接原因之一。

第三，**memory access pattern 不够友好**。  
Winograd 的执行流程包含：

- 对输入进行 tile 划分；
- 为每个 tile 构造局部 `4x4` 数组；
- 对中间矩阵 `U`、`V`、`M` 做频繁访问；
- 在不同阶段切换数据布局。

这种访问模式相比直接卷积更复杂。当前代码是教学实现，没有专门优化数据布局、缓存复用和寄存器 blocking，因此容易产生额外的内存访问成本。换句话说，虽然算术乘法减少了，但 memory traffic 和数据搬运增加了。

第四，**baseline 本身已经很轻量**。  
本实验直接卷积的输入规模仅为 `56 x 56 x 3`，输出通道为 64，而且 batch 为 1。这个规模下，直接卷积的三重循环其实并不重，实测时间只有约 `5.38e-4` 秒。当 baseline 足够小且简单时，Winograd 很难仅靠减少乘法数量就赢回来。

第五，**当前实现没有使用高性能优化手段**。  
工业级 Winograd 实现通常会结合：

- SIMD 向量化；
- 多线程并行；
- 更紧凑的 transform 融合；
- 更高效的 tile/layout 设计；
- 与卷积 kernel 深度配合的寄存器级优化。

而本实验实现只是正确地复现了算法流程，并未做上述优化。因此，当前结果更接近“算法验证版”的表现，而不是“高性能 kernel 版”的表现。

综合来看，当前 Winograd **没有提升性能，反而更慢**，这是一个合理结果。它说明理论上的 FLOPs reduction 需要配合良好的 memory access pattern 与 transform optimization，才能真正转化为实测加速。

## 3. Correctness

### 3.1 Strassen Correctness

Strassen 最大相对误差小于 `1e-3` 判定阈值：

| Matrix Size | Max Relative Error | Result |
|---|---:|---:|
| 256 x 256 | 2.75139e-06 | PASS |
| 512 x 512 | 5.62508e-06 | PASS |
| 1024 x 1024 | 1.31208e-05 | PASS |

### 3.2 Winograd Correctness

Winograd 最大绝对误差小于 `1e-1` 阈值：

| Method | Max Error | Result |
|---|---:|---:|
| Winograd vs. Direct Convolution | 6.10352e-05 | PASS |

故以上检查表明程序实现正确

