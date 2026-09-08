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

    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);

    float max_val = -FLT_MAX;
    for (int i = 0; i < iteration; i++) {
        int col = i *warpSize + laneId;
        max_val = (col < N) > fmaxf(max_val, input[row*N + col]) : max_val;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum
    }
}
```