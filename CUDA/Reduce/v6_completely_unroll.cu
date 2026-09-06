#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#define THREAD_PER_BLOCK 256

template <int blockSize>
__device__ void warpReduce(volatile float* cache, int tid) {
    if (blockSize >= 64) cache[tid] += cache[tid + 32];
    if (blockSize >= 32) cache[tid] += cache[tid + 16];
    if (blockSize >= 16) cache[tid] += cache[tid + 8];
    if (blockSize >= 8) cache[tid] += cache[tid + 4];
    if (blockSize >= 4) cache[tid] += cache[tid + 2];
    if (blockSize >= 2) cache[tid] += cache[tid + 1];
}

template <int blockSize>
__global__ void reduce_v6(float* d_input, float* d_output) {
    __shared__ float s_data[THREAD_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    s_data[threadIdx.x] = d_input[idx] + d_input[idx + blockDim.x];
    __syncthreads();

    if (blockSize >= 512) {
        if (threadIdx.x < 256) s_data[threadIdx.x] += s_data[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIdx.x < 128) s_data[threadIdx.x] += s_data[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64) s_data[threadIdx.x] += s_data[threadIdx.x + 64];
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce<blockSize>(s_data, threadIdx.x);
    if (threadIdx.x == 0) d_output[blockIdx.x] = s_data[0];
}

// __global__ void reduce_v0(float* d_input, float* d_output) {
//     __shared__ float s_data[THREAD_PER_BLOCK];
//     float* input_begin = d_input + blockIdx.x * THREAD_PER_BLOCK;
//     s_data[threadIdx.x] = input_begin[threadIdx.x];
//     __syncthreads();

//     for (int i = 1; i < THREAD_PER_BLOCK; i *= 2) {
//         if (threadIdx.x % (i * 2) == 0)  {
//             s_data[threadIdx.x] += s_data[threadIdx.x + i];
//         }
//         __syncthreads();
//     }
//     if (threadIdx.x == 0) {
//         d_output[blockIdx.x] = s_data[0];
//     }
// }

bool check(float* out, float* res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float* input = (float*)malloc(N * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = N / (THREAD_PER_BLOCK * 2);
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    float* result = (float*)malloc(block_num * sizeof(float));
    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)rand() / RAND_MAX;
    }
    // cpu calculation: each block now reduces THREAD_PER_BLOCK * 2 elements
    const int elems_per_block = THREAD_PER_BLOCK * 2;
    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < elems_per_block; j++) {
            cur += input[i * elems_per_block + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce_v6<THREAD_PER_BLOCK><<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("Test passed\n");
    } else {
        printf("result: %f, expect: %f\n", output[0], result[0]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);
    return 0;
}