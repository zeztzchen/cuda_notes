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

naive 版：向全局内存中写数据，使用原子函数会导致线程变成序列化，丧失并行性，算子性能降低
```cpp
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v1<<<grid_size, block_size>>>(d_x, d_y, N);

__global__ void reduce_v1(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) atomicAdd(output, input[idx]);
}
```

折半归约：在block内进行归约，每个block归约一部分到block内的shared memory中，然后归约到第一个元素中

```cpp
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v2<<<grid_size, block_size>>>(d_x, d_y, N);

__global__ void reduce_v2(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    __shared__ flaot input_s[BLOCK_SIZE];

    // 1. 搬运和线程数（blockDim.x）相等的数据，到block的共享内存中
    input_s[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 2. 用1/2, 1/4, 1/8...的线程进行折半归约
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            input_s[tid] += input_s[tid + offset];
        }
        __syncthreads();
    }

    // 3. 每个block的第一个线程将计算结果累加到输出中
    if (tid == 0) atomicAdd(output, input_s[0]);
}
```

**warp shuffle（推荐写法）**: 在 warp 内进行折半归约，其优势在于，一个 warp 内的线程是同步的，相比于以 block 为单位进行折半，以 warp 为单位进行每次折半时不需要 __syncthreads()，并行性更高。
> BLOCK_SIZE需要是32的整数倍，否则产生线程数不足32的warp，可能会导致访问到无效数据。

个人理解：这里warp版本和之前的区别在于，前面是一个block上进行计算，这里是一个block中每个warp算好，然后汇总，然后再汇总一次

```cpp
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v3<<<grid_size, block_size>>>(d_x, d_y, N);

__global__ void reduce_v3(float* d_x. float* d_y, N) {
    __shared__ float s_y[32]; // 仅需要32个，因为一个block最多1024个线程，最多1024/32=32个warp

    int idx = blockDimx. * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize; // 当前线程属于哪个warp
    int laneId = threadIdx.x % warpSize; // 当前线程是warp中的第几个线程

    float val = (idx < N) d_x[idx] : 0.0f; // 搬运d_x[idx]到当前线程的寄存器中
    #program unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset); // 在一个warp里折半归约
    }

    if (laneId == 0) s_y[warpId] = val; // 每个warp里的第一个线程，负责将数据存储到shared mem中
    __syncthreads();

    if (warpId == 0) { // 使用每个block中的第一个warp对s_y进行最后的归约
        int warpNum = blockDim.x / warpSize; // 每个block中的warp数量
        val = (laneId < warpNum) ? s_y[laneId] : 0,0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl__down__sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0) atomicAdd(d_y, val); // 使用此warp中的第一个线程，将结果累加到输出
    }
}
```

warp shuffle + float4: 在 warp shuffle 上进一步优化，搬运数据时使用 float4：
```cpp
#define FLOAT4(value) (float4*)(&(value))[0]
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(CEIL(N, BLOCK_SIZE),4));  // 这里要除以4
reduce_v3<<<grid_size, block_size>>>(d_x, d_y, N)

__global__ void reduce_v4(float* d_x, float* d_y, const int N) {
    __shared__ float s_y[32];
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // 这里要乘以4
    int warpId = threadIdx.x / warpSize;   // 当前线程位于第几个warp
    int laneId = threadIdx.x % warpSize;   // 当前线程是warp中的第几个线程
    float val = 0.0f;
    if (idx < N) {
        float4 tmp_x = FLOAT4(d_x[idx]);
        val += tmp_x.x;
        val += tmp_x.y;
        val += tmp_x.z;
        val += tmp_x.w;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (landId == 0) atomicAdd(d_y, val);
    }
}
```

## Triton 篇


## FlashAttention 专项篇


## Cutile 篇