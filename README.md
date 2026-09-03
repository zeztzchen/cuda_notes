# CUDA Notes

## 引言

快速复习 CUDA 面试，本仓库包含 CUDA、Triton、Cutile三部分内容

## 基础知识



## CUDA 篇

参考内容：https://github.com/Tongkaio/CUDA_Kernel_Samples

```cpp
// 向上取整
#define CEIL(a, b) ((a+b-1)/(b))

// FLOAT4，用于向量化访存，以下两种都可以
#define FLOAT4(value) *(float4*)(&(value))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
```

### elementwise

简介：逐元素操作
写法：
1. naive：逐个元素操作
2. float4等向量化访存方法，要在grid上除以4

#### add
naive 版
```cpp
// block_size，grid_size 和函数调用
int block_size = 1024;
int grid_size = CEIL(N, block_size);
elementwise_add<<<grid, block_size>>>(a, b, c, N);

__global__ void elementwise_add(float *a, float *b, float *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```
向量化访存版
```cpp
#define FLOAT4(value) *(float4*)(&(value))

int block_size = 1024
int grid_size = CEIL(CEIL(N,4), block_size);
elementwise_add_float4(a, b, c, N);

__global__ void elementwise_add_float4(float *a, float *b, float *c, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;

        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.x = tmp_a.x + tmp_b.x;

        FLOAT4(c[idx]) = tmp_c;
    }
}
```

#### sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

```cpp
__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void sigmoid_float4(float* x, float* y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_x.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_x.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_x.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_x.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FLOAT4(y) = tmp_y;
    }
}
```

#### relu

```cpp
__global__ void relu(float *x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_float4(float *x, float* y, int N) {
    int idx = (blockDim.x * blockSize.x + threadIdx.x) * 4;
    if (idx < N) {
        
    }
}
```
### reduce


## Triton 篇


## FlashAttention 专项篇


## Cutile 篇