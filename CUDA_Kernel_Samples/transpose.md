## transpose

矩阵转置，考察GPU全局内存高校访问、bank conflict 知识点

如何优化全局内存的访问：
1. 尽量合并访问，连续的线程读取连续的内存，尽量让访问的全局内存的首地址是32字节的倍数（一次性数据传输的数据量）
2. 如果不能同时合并读取和写入，则应该尽量做到合并写入。因为编译器如果能判断一个全局内存变量在核函数内是只可读的，会自动调用 __ldg() 读取全局内存，从而对数据进行缓存，缓解非合并访问带来的影响，但这只对读取有效，写入则没有类似的函数。另外，对于开普勒架构和麦克斯韦架构，需要显式的使用 __ldg() 函数，例如 B[ny * N + nx] = __ldg(&A[nx * N + ny])。


naive
```cpp
__global__ void transpose(float* input, float* output, int M, int N) {
    // input 的 row 和 col
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

```

仅合并写入
```cpp
__global__ void transpose(float* input, float* output, int M, int N) {
    // output 的 row 和 col
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col * N + row]); // 合并写入，读取使用__ldg进行缓存
    }
}

```

推荐：使用共享内存中转，同时合并读取和写入

需要注意的是，这种方式在读共享内存数据时会遇到经典的 bank conflict 问题，可通过 padding 或者 swizzling 的方式解决：
对共享内存做padding：
```cpp
dim3 block(32, 32);
dim3 grid(CEIL(N, 32), CEIL(M, 32));
transpose<32><<<grid_size, block>>>(input, output, M, N) {
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE+1]; // padding
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x1 = bx + threadIdx.y;
    int y1 = by + threadIdx.x;

    if (x1 < N && y1 < N) {
        s_mem[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if (x2 < M && y2 < N) {
        output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.y];  // padding后，此处不存在bank conflict
    }
}
```

使用 swizzling，不需要对共享内存做 padding：
```cpp
// 输入矩阵是M行N列，输出矩阵是N行M列
dim3 block(32, 32);
dim3 grid(CEIL(N,32), CEIL(M,32));  // 根据input的形状(M行N列)进行切块
transpose<32><<<grid, block>>>(input, output, M, N);

template <const int BLOCK_SIZE>
__global__ void transpose(float* input, float* output, int M, int N) {
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE];  // 不需要padding
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;

    if (x1 < N && y1 < M) {
        s_mem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
    }
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if (x2 < M && y2 < N) {
        output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.x ^ threadIdx.y];  // swizzling后，此处不存在bank conflict
    }
}
```