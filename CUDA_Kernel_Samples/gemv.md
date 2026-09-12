## gemv

矩阵乘以一个向量，方法：每个block有一个warp，每一个warp负责一行的计算。

```cpp
// 行数：M = 1024
// 列数：K = 32
// block 的数量和行数相同：grid_siz = M
// 每个block里有一个warp：block_size = 32
sgemv<<<grid_size, block_size>>>(A, x, y, M, K);
__global__ void sgemv(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if (row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K, warpSize);

    for (int i = 0; i < KIteration; i++) {
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if (laneId == 0) y[row] = res;
}
```