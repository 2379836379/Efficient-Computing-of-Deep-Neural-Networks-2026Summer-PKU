# Lab 1: GEMM 课件翻译

以下内容是 `lab1_slides.pdf` 的中文整理版。

## 1. 标题页

CMSC5743 实验 01：GEMM  
Bei Yu  
香港中文大学 计算机科学与工程系  
byu@cse.cuhk.edu.hk  
2026 年 7 月 8 日

## 2. Memory Hierarchy

存储层次结构主要分为三类：

- Cache Memory，缓存
- Primary Memory / Main Memory，主存
- Secondary Memory，辅存

### Cache Memory

- 缓存比主存更快
- 访问延迟比主存更低
- 用来存放短时间内会执行的程序
- 也用于临时数据存储

### 但缓存也有缺点

- 容量有限
- 成本很高

### Primary Memory / Main Memory

- 通常是易失性存储器
- 是计算机的工作内存
- 比辅存更快
- 计算机没有主存就无法运行

## 3. Cache Performance

- 如果处理器发现某个内存位置已经在 cache 中，这叫 `cache hit`，数据直接从 cache 中读取
- 如果处理器没有在 cache 中找到该位置，这叫 `cache miss`
- 发生 `cache miss` 时，cache 会分配新条目，并从主存把数据拷贝进来，然后再用 cache 中的数据完成请求
- 命中率公式：`hit / (hit + miss)`，也就是命中次数 / 总访问次数

## 4. Data Layout

这一页主要在说明矩阵和张量的数据布局示意图。

## 5. Matrix Multiplication

这一页是矩阵乘法示意图。

## 6. 使用转置

- 如果我们使用转置来改变矩阵的访问顺序，会怎样？
- 命中率会有什么差别？

## 7. 循环顺序

- 循环顺序，也就是 dataflow，可能会影响性能
- 可以思考使用更多优化方法来提高 cache 命中率

## 8. 输出矩阵 C 的访问模式

对输出矩阵 C 的每个元素来说：

- Inner：对 A、B 读取 K 次，对 C 写入 1 次
- Outer：对 A、B 只读取 1 次，但对 C 写入 K 次（按 frame）
- Row-based：对 A 只读取 1 次，对 B 读取 K 次，对 C 写入 K 次（按 row）

## 9. 总结

这份课件的核心是在说明：

- GEMM 的性能不只取决于算法复杂度
- 更关键的是内存访问模式是否符合 cache 规律
- 转置、循环重排和其他优化都可能显著影响实际运行时间

## 10. 备注

课件里有几页主要是图示，没有额外文字说明，所以这里只翻译了可提取的文字内容。
