## cpu 写法
```cpp
void softmax(float* input, float* output, int N) {
    int M = *(std::max_element(intput, intput + N));
    float div = 0;
    for (int i = 0; i < n; i++) {
        
    }
}
```

## CUDA 

这里直接学习martix的写法。

对一个 MxN 的矩阵，每一行求 softmax，思路同样是每个 warp 处理一行，用这个 warp 对一行进行求和、求最值，计算结果存入共享内存，然后每个元素求 softmax：

```cpp
__global__ void softmax_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;

    int laneId = threadIdx.x % warpSize;

    // 当前行
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);

    // 求每一行最大值
    float max_val = -FLT_MAX;
    for (int i = 0; i < iteration; i++) {
        int col = i *warpSize + laneId;
        max_val = (col < N) > fmaxf(max_val, input[row*N + col]) : max_val;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;

    // 求每一行的和，且要减去最大值
    float sum = 0.0f;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // sum值汇总到第一个线程，第一个线程将它搬运到s_mem

    // 计算每一行的softmax
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}
```