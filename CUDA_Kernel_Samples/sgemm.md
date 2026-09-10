# sgemm

矩阵乘

## naive
```cpp
// C(MxN) = A(MxK) * B(KxN) 行优先
// 每个线程处理一个输出矩阵中的元素

// 假设 M N K 已经赋值

const int BLOCK_SIZE = 32;
dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float accum = 0.0f;
    for (int i = 0; i < K; i++) {
        accum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = accum;
}
```

## block_tile 版本

```cpp
#define BLOCK_SIZE 32
dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(CEIL(N,BLOCK_SIZE), CEIL(M,BLOCK_SIZE));  // 根据C矩阵的形状(M行N列)切块
sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= M || idy >= N) return;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 初始化 block tile 配置
    A = &A[(by * BM) * K];
    B = &B[bx * BN];
    C = &C[(by * BM) * N + bx * BN];

    float accum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        // 搬运 global ==> shared
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();
        A = A + BK;
        B = B + BK * N;
        for (int i = 0; i < BK; i++) {
            accum += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads();
    }

    C[ty * N + tx] = accum;
}
```

## thread_tile
一个线程承担更多的计算，更加高效：

```cpp
dim3 block(256);
dim3 grid(CEIL(N,128), CEIL(M,128));  // 根据C矩阵的形状(M行N列)切块
sgemm<128, 128, 8, 8, 8><<<grid, block>>>(A,B,C,M,N,K);

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdy.y;

    int block_row_thread = BN / TN;  // block中一行的thread数量
    int block_col_thread = BM / TM;  // block中一列的thread数量
    int thread_num = block_row_thread * block_col_thread;  // block中thread总量

    int tx = (threadIdx.x % block_row_thread) * TN;  // threadtile左上角x坐标
    int ty = (threadIdx.x / block_row_thread) * TM;  // threadtile左上角y坐标

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // BM/(BM/(thread_num/BK)) = thread_num/BK = stride

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float accum[TM][TN] = {0.0f};
    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int row = 0; row < TM; row++) {
            for (int col = 0; col < TN; col++) {
                for (int i = 0; i < BK; i++) {
                    accum[row][col] += As[(ty + row) * BK + i] * Bs[i * BN + (tx + col)];
                }
            }
        }
        __syncthreads();
    }
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }
}
```