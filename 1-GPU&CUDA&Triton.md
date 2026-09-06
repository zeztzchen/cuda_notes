## 达标目标

- 能解释 GPU → SM → warp → thread 的执行层次。
- 能解释 register、shared memory、L1/L2、HBM 的容量、延迟与作用域。
- 能分析 coalesced access、bank conflict、warp divergence、occupancy 和 latency hiding。
- 能用 Roofline 判断算子是计算受限还是访存受限：

$$
AI = \frac{\text{FLOPs}}{\text{Bytes}}, \qquad
P \leq \min(P_{\text{peak}}, BW \times AI)
$$

- 能写并优化 Reduce、Softmax、GEMM 的基本 CUDA/Triton kernel。
- 能解释 Tensor Core 的 MMA 运算与 CUDA Core 的区别。
- 能推导 Online Softmax，并手写 FlashAttention forward 的分块框架。
- Python 达到熟练使用 PyTorch/Triton、测试和 benchmark 的水平；C++ 达到能够阅读、修改 CUDA kernel 和简单 PyTorch CUDA Extension 的水平。


## 需要学习的资料

https://github.com/Tongkaio/CUDA_Kernel_Samples/tree/master 完全够用了！
- reduce
- sgemm

flash attention
- 


## 学习笔记

### Sgemm

优化路径

global memory -> shared memory -> register

简单来说就是将global内加载到shared中，进一步加载到register中，减少从慢的设备中的访问加快读写

读：2mnk
写：mn

优化后：

读：
写：