GPU 中有专门用来进行数学计算的

性能指标：
- 核心数
- GPU显存容量
- GPU计算峰值
- 显存带宽

GPU不能单独计算，CPU + GPU组成异构计算架构
- CPU控制（Host）
- GPU看作CPU的协处理器，称为设备（Device）
内存之间的访问通过PCIe总线连接（PCIe是很慢的）

运行时API：
- 驱动 driver API （汇编？）
- 运行时 runtime API （更加友好，C++）

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250417211724220.png)

核函数，在GPU中进行，会打印16个hello
```cpp
#include <stdio.h>

__global__ void kernel() {
    printf("Hello from the kernel!\n");
}

int main() {
    kernel<<<4, 4>>>(); // 4*4个线程
    cudaDeviceSynchronize(); // 同步代码 Wait for the kernel to finish
    return 0;
}
// Compile with: nvcc -o test test.cu
```

# 核函数

调用gpu设备
> gpu 是 cpu 的外设，需要cpu进行指令控制
> 主机对设备的调用通过核函数
- 核函数在GPU上进行并行执行
- 注意：
	- 限定词 `__global__` 修饰
	- 返回值必须是 void
- 形式：位置可以互换
```cpp
__global__ void kernel_function

void __global__ kernel_function
```

注意事项：
- 核函数只能访问GPU内存（CPU和GPU通过PCIe总线连接，运行时API特定函数完成交互）
- 核函数不能使用变长参数（要明确个数）
- 核函数不能使用静态变量
- 核函数不能使用函数指针
- 核函数具有异步性（和CPU GPU异构架构有关，使用核函数的时候只是启动了核函数，但是CPU主机无法控制GPU设备的执行，CPU主机不会等待GPU执行完毕，需要显示调用同步函数，同步主机CPU与设备GPU的工作进程，有些线程的执行也需要同步）

编写流程
核函数不支持iostream（使用printf）

```cpp
int main(void)
{
	主机代码
	核函数调用
	主机代码
	return 0；
}
```

`<<a, b>>` a 线程块的个数，b 每个线程块中线程的数量
`cudaDeviceSynvhronize();` 等待GPU执行完毕（处理好主机和设备的同步）

# 线程模型

## 线程模型结构

重要概念：
- grid 网格（各核函数启动所产生的所有线程统称为一个网格grid，一个核函数对应一个grid，其中包含若干个线程块）
- block 线程块（包含若干个线程 Thread GPU编程最小单位）
> 核函数是主机代码中调用，主机端启用之后，在GPU上执行，每个线程都会执行核函数

线程分块是逻辑上的划分

配置线程：`<<<grid_size, block_size>>>`

最大允许线程块大小：1024
最大允许网格大小：2^31 - 1 （针对一维网格）

> 线程个数远远高于计算核心数量（几百上千），设计==总线程数== 至少等于==计算核心数== 实际上要大于 GPU计算的时候内存访问同时进行，减少空闲时间，重叠CPU和GPU的运行，因为核函数是启动GPU计算，可以先做别的事情

## 一维线程模型

每个线程在核函数都有唯一标识

唯一标识由 `<<<grid_size, block_size>>>` 确定，保存在内键变量（一维）不用定义，可以直接使用
- gridDIM.x：等于执行配置中变量grid_size的值
- blockDIM.x：等于执行配置中变量block_size的值

线程索引保存成内建变量
- blockIdx.x：线程在网格中线程块的索引 0 - gridDIM.x -1
- threadIdx.x：线程在线程块中的索引 0 - blockDIM.x - 1

线程唯一表示：`Idx = threadIdx.x + blockIdx.x * blockDim.x`

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419140043132.png)

## 推广到多维

CUDA可以组织三维的网格和线程块

blockIdx和threadIdx类型uint3的变量，该类型是一个结构体，具有x,y,z三个成员（无符号类型成员）

gridDim和blockDim是类型为dim3的变量，是一个结构体，具有x,y,z

取值范围 0 - gridDim.x(y/z)-1

注意：内建变量只在核函数中有效，且无需定义！

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419140800461.png)


![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419140921545.png)


排列和矩阵不一样
- 最先变化x，所以每一行是x在改变
- 本质上还是一维
- tid 是线程在线程块中唯一索引，bid 是网格中线程块的索引
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419140958137.png)


# 线程全局索引计算方式


一维 `int id = threadIdx.x + blockIdx.x * blockDim.x`
二维

`int blockId = blockIdx.x + blockId.y * gridDim.x`
`int threadId = threadIdx.y * blockDim.x + threadIdx.x`
`int id = blockId * (blockDim.x * blockDim.y) + threadId`

# nvcc编译流程与GPU计算能力

nvcc编辑 虚拟架构兼容性 真实架构兼容性
GPU不同不一定可以执行（可移植性）

## nvcc编译流程

分离全部源代码
- 主机代码 C++
- 设备代码 C++扩展语言

nvcc将设备代码编译为PTX伪汇编代码，再编译为二进制cubin目标代码

将源代码编译为PTX代码时，需要用 `-arch=compute_XY` 指定一个虚拟架构的计算能力，用以确定代码中能够使用的CUDA功能

将PTX代码编译为cunbin代码时，需要用 `-code=sm_ZW` 指定真实架构的计算能力（大于虚拟），用以确定可执行文件能够使用的GPU


## PTX

增加可移植性，中间层
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419143031230.png)

## GPU架构与计算能力

二进制兼容性不一定可以跨架构应用 指令集指令编码不一定相同

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419143434599.png)
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419143520926.png)

# CUDA程序兼容性问题

移植问题

虚拟架构计算能力 `-arc=compute_61` 只能在计算能力大于等于6.1的GPU上执行

实际架构计算能力 `-code=sm_XY`
- 与具体的GPU架构有关
- 大版本之间兼容
- 必须指定虚拟架构且真实大于等于虚拟架构
可以实现低小版本到高小版本的兼容

# 指定多GPU版本编译


![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419144514896.png)

## 即时编译

ptx 嵌入可执行文件中

安培架构 8
帕斯卡架构 6
电脑直接编译不能再安培架构中运行，设置即时编译，可执行文件中嵌入PTX代码，安培架构中执行可执行文件的时候，嵌入的代码可以直接在GPU中运行，缺点是无法发挥性能，在安培中运行时，会根据虚拟架构PTX代码即时编译一个适用安培架构的代码
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419144749952.png)

## 默认计算能力

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419144952604.png)

# CUDA 矩阵加法运算程序

## CUDA 程序基本框架

```cpp
# include <头文件>

__global__ void 函数名(参数) {
	核函数内容
}

int main() {
	设置GPU设备
	分配主机和设备内存 // 分别进行内存分配
	初始化主机中的数据 
	数据从主机复制到设备 // 通过运行时API，核函数只能用显存中的数据
	调用核函数在设备中进行计算 
	将计算结果得到的数据从设备传到主机 // 通过PCIe组件 慢
	释放主机与设备内存
    return 0;
}


```


例子：
```cpp
#include <stdio.h>
#include "../common.cuh"

__global__ void addFromGPU(float *fpA, float *fpB, float *fpC, const int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    fpC[i] = fpA[i] + fpB[i];
}

void initialData(float *addr, int elemCount) {
    for (int i = 0; i < elemCount; i++) {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main() {
    setGPU();

    int iElemCount = 512;
    size_t stBytesCount = iElemCount * sizeof(int);

    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    if (fpHost_A == NULL || fpHost_B == NULL || fpHost_C == NULL) {
        printf("Error allocating host memory\n");
        exit(-1);
    }
    else {
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }

    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float **)&fpDevice_A, stBytesCount);
    cudaMalloc((float **)&fpDevice_B, stBytesCount);
    cudaMalloc((float **)&fpDevice_C, stBytesCount);

    if (fpDevice_A == NULL || fpDevice_B == NULL || fpDevice_C == NULL) {
        printf("Error allocating device memory\n");
        exit(-1);
    }
    else {
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }

    srand(42);
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid(iElemCount / 32);

    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
    cudaDeviceSynchronize();

    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();

    printf("Done\n");
    return 0;
}
```
## 设置GPU设备

获取GPU设备数量
```cpp
int iDeviceCount = 0;
cudaGetDeviceCount(&iDeviceCount);
```

设置GPU执行时使用的设备
```cpp
int iDev = 0;
cudaSetDeviceCount(iDev);
```

## 内存管理

CUDA 通过内存分配、数据传递、内存初始化、内存释放进行内存管理

malloc cudaMalloc
memcpy cudaMemcpy
memset cudaMemset
free   cudaFree

### 内存分配

主机和设备都可以调用
双重指针：C不支持多个返回值，这里需要返回错误代码，所以利用双重指针返回地址（进行地址的分配）
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419152125219.png)

### 数据拷贝

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419152335205.png)

### 内存初始化

按字节初始化

目的：分配地址不初始化，内存地址无意义，访问的时候可能报错带来崩溃

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419152421342.png)

### 内存释放

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419152615590.png)


## 自定义设备函数

设备函数（device）
- 修饰符 `__device__`
- 只能被核函数和其他设备函数调用，定义只能执行在GPU设备上的函数

核函数（kernel）
- 修饰符 `__global__`
- 主机调用，设备执行
- 不能与host和device同时使用

主机函数（host）
- 修饰符 `__host__`，对于主机端函数可省略
- 可以用host和device同时修饰，减少冗余，编译器会针对主机和设备分别编译该函数
# CUDA 错误检查

## 运行时API错误代码

返回值类型：
- cudaError_t

成功：cudaSuccess

运行时API返回的执行状态是枚举变量（enumError）
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419214515466.png)


## 错误检查函数

可以在主机和设备中执行(`__host__ __device__`)

获取错误代码对应名称：cudaGetErrorName
- 返回字符串类型指针 char*
- 枚举变量名称

获取错误代码描述信息：cudaGetErrorString
- 返回字符串类型指针 char*
- 描述信息

## 错误检查函数（不能捕捉核函数）

调用CUDA运行时API时，调用ErrorCheck函数进行包装

参数filename一般用 `__FILE__` 参数 lineNumber 一般使用 `__LINE__`

错误函数返回运行时API调用的错误代码

```cpp

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber) {
	if(error_code != cudaSuccess) {
		... // 打印信息 错误 名称 文件 行号
		return error_code;
	}
	return error_code;
}

```

## 检查核函数

核函数只会返回 void

错误检测函数问题：不能捕捉调用核函数的相关错误

捕捉调用核函数可能发生错误的方法：
`ErrorCheck(cudaGetLastError(), __FILE, __LINE_)` 检测同步之上的最后一个错误
`ErrorCheck(cudaDeviceSynchronize(), __FILE, __LINE_)` CPU和GPU是异步结构，调用核函数之后主机会运行之后的代码，所以需要运行同步函数同步主机与设备

# CUDA 记时

## 事件记时

- 程序执行时间记时：CUDA程序执行性能的重要表现
- 使用CUDA事件（event）记时方式
- CUDA事件记时可为主机代码、设备代码记时

> 因为 cudaEventQuery API大概率返回一个错误，但是这不代表程序出错了
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419222111261.png)


## nvprof 性能刨析

这是一个可执行文件

执行命令：`nvprof ./可执行文件`

# 运行时GPU信息查询

## API

涉及的API

调用：
- `cudaDeviceProp prop` 结构体变量
- `ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__)` 

## 查询GPU计算核心数量

CUDA 运行时API函数无法查询GPU核心数量

根据GPU的计算能力进行查询 mp（流处理器数量） x 128 （和架构相关）

# 组织线程模型

线程模型与多维数组（二维）的一一对应关系

线程确定唯一的索引进行并行的运算

数据存储方式
- 16 x 8：16列 8行（以前都是先行后列）
> 下图标记错误，上面的row是col，下面的nx=16为ny=8
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419224447515.png)


想要高效处理
- 
## 二维网格二维线程块

发挥多线程优势：每个线程处理不同数据
高效并行：分配每个线程，每个线程处理不同数据
避免多个不同线程处理同一个数据，没有组织的胡乱访问内存

128 个线程 8 个block

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419225155133.png)

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419225340729.png)

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419225625589.png)


![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419231429858.png)

设计循环来进行并行计算
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419231551617.png)
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250419231618827.png)

# GPU 硬件资源

GPU并行性依靠流多处理器——SM（streaming muliprocessor）

一个GPU由多个SM构成，Fermi架构SM关键资源如下：
1. CUDA核心（CUDA core） ARU 整型算术逻辑单元 FPU 浮点数算数单元
2. 共享内存/L1缓存 大小可以通过运行时API配置（整个线程块共享）
3. 寄存器文件 保存和寄存器相关内容
4. 加载和存储单元 （Load/Store Units）
5. 特殊函数单元 SFU 高效函数
6. Warps调度（Warps Scheduler）两个线程束调度器 两个指令调度单元

并行：同时执行没有干扰
并发：一个核心，任务高效切换，同一个时间节点只做了一件事

- GPU每个SM都可以支持数百个线程==并发==执行
- 以线程块block为单位，向SM分配线程块，多个线程块分配到同一个SM上
- 一个线程块被分配好SM后，不可以分配到其他SM上   

## 线程模型和物理结构

线程模型：
- 逻辑角度
- 可以定义成千上万个线程
- 所有线程块block都需要分配到SM上执行
- 线程块内所有线程分配到同一个SM中执行，==每个SM可以被分配多个线程块==
- 线程块分配到SM中后，会以32（Warp 线程束）个线程为一组进行分割（这32个是并行的）

物理结构：
- 硬件资源是有限的，活跃的线程束的数量会受到SM资源限制
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250420102044143.png)

## 线程束

CUDA采用单指令多线程SIMT架构管理执行线程，每32个一组，构成一个线程束
同一个线程块中相邻的32个线程构成一个线程束（同一个线程块中只有这32个并行）

每个线程束中只包含同一线程块中的线程
线程束是GPU硬件上真正做到了并行

` 线程数 = ceil(线程块中的线程数 / 32) `
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250420102353252.png)

# CUDA 内存模型概述

## 结构层次特点

GPU执行任务时需要反复加载存储，访问内存数据是制约速度的因素

局部原则性：
- 时间局部性：一个数据被访问，可能会被再次访问，随时间降低
- 空间局部性：一个地址被访问，可能再次被访问，随距离降低

底部存储器特点：
- 更低的每比特位平均成本
- 更高的容量
- 更高的延迟
- 更低的处理器访问频率

CPU和GPU主存采用DRAM（动态随机存取存储器）
低延迟的内存采样SRAM（静态随机存取存储器）

寄存器最快
主存 8G 16G 运行内存
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250420102848230.png)

## 内存模型

线程块有共享内存，给一个线程块中的线程进行数据交换

全局内存：所有线程共享

线程可以读取常量和纹理内存，不能修改
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250420103322967.png)
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250420103631499.png)

# 寄存器和本地内存（局部内存）

## 寄存器

片上（on-chip）
生命周期与线程一致，仅线程内可见
未添加限定符（`__share__`）等的变量一般存放在寄存器中
内建变量存在寄存器中，gridDim blockDim blockIdx

不加限定符的数组可能在寄存器也可能在本地内存中（无法保存过多数据）

寄存器都是32位，寄存器保存在SM的寄存器中（存duoble需要2个寄存器）
计算能力5.0-9.0的GPU，每个SM中都是64k的计算器（寄存器）数量（Fermi架构32K）
每个线程最大寄存器数量是==255==个，Fermi架构是63个

## 本地内存（局部内存）

寄存器放不下
- 比较大的数组/结构体，占用大量资源的
- 索引值不能编译时确定的数组
- 任何不满足寄存器保存条件的变量

每个线程最高512KB本地内存
硬件角度是全局内存一部分，延迟高
对于计算能力2.0以上的设备，本地内存的数据存储在每个SM的一级缓存和设备的二级缓存中
## 寄存器溢出

核函数所需寄存器数量超过硬件设备支持，数据保存到本地内存
- 一个SM运行并行运行多个线程块/线程束，总的需求寄存器大于64KB
- 单个线程运行所需寄存器数量255个

寄存器溢出会降低运行性能
- 本地内存是全局内存一部分，延迟高
- 寄存器溢出的部分可以进入GPU缓存中

# 全局内存

存在片外
特点：容量大、延迟高、使用多

全局内存的数据所有线程可见，Host端可见，且具有与程序相同的生命周期（cudaMalloc开始 cudaFree结束）

初始化：
- 动态：cudaMalloc
- 静态：`__device__`编译器编译期间确定，外部定义，核函数可以直接访问不用参数传递，主机中需要通过API cudaMemcpyToSymbol cudaMemcpyFromSymbol
`__device__ ind d_x = 1;`


## 使用cmake管理


# 共享内存

- 片上（on-chip）单位是KB(全局是GB),有更高的带宽和更低的延迟
- 线程块内所有线程可见，生命周期与线程块一致
- `__share__` 修饰的变量放在共享内存中，有静态动态两种
- 每个SM的共享内存数量是一定的，单个线程块分配过度的共享内存会限制活跃线程束的数量
- 访问共享内存必须加入同步机制：线程块内同步 `void __syncthreads();`


不同计算能力的架构，每个SM拥有的共享内存大小不同
计算能力8.9是每个SM有100K（每个线程块最大99kb）

作用：可以被程序员直接操控的缓存
减少核函数中对全局内存的访问次数，实现高效线程块内部通信
- 经常访问的数据由全局内存搬移到共享内存，提高访问效率
- 改变全局内存访问内存的内存事务方式，提高数据访问的带宽

## 静态共享内存

`__shared__`

`__shared__ float tile[size, size];`

作用域：
- 核函数中声明，静态共享内存作用域局限在这个核函数中
- 文件核函数外声明，静态共享内存作用域对所有核函数有效

编译时要确定内存大小

## 动态共享内存

`extern __shared__ `

`extern __shared__ float title[];` 不能指定大小

运行的时候要在核函数中指定第三个参数，为内存的大小 `<<<grid, block, 32>>>`
# 常量内存

有常量缓存的全局内存，数量有限，大小64KB，由于有缓存，访问速度比全局内存快

对同一编译单元所有线程可见

使用 `__constant__` 修饰，不能定义在核函数中，静态定义

常量内存仅可读不可写

核函数传递数值参数时，变量就存在常量内存中

- 定义时初始化
- 运行时初始化方法：主机端使用cudaMemcpyToSymbol初始化
- 线程束中所有线程需要从相同内存地址读取数据，常量内存表现最好。只需要读取一次传播给所有线程。例如：数学公式中的系数
# GPU缓存

GPU缓存是不可编程内存

每个SM都有一个一级缓存，所有SM共享一个二级缓存（L2 Cache）

L1缓存和L2缓存用来存储本地内存和全局内存的数据，也包括寄存器溢出的部分

在GPU上只有内存加载可以被缓存，内存存储操作不能被缓存

每个SM有一个只读常量缓存和只读纹理缓存，用于设备内存中提高各自内存空间的读取性能 

全局内存加载流程
- 全局内存是逻辑上的划分，数据存储在DRAM上
- 加载全局内存的数据要经过L2 Cache，可能还需要经过L1 Cache（可以编译选项设置），最后才会被流处理器运算
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424101608657.png)

计算能力8.9显卡为例：
- 统一数据缓存大小为128KB，包括==共享内存、纹理内存、L1缓存==
- 可以配置，但不一定生效，GPU会自动最优
- 共享内存从统一数据缓存分区，可以配置各种大小0 8 16 64 100，剩下的用作L1缓存和纹理单元使用
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424102607849.png)
# 计算资源分配

（线程束真正做到了并行）
线程束本地执行上下文主要资源组成：
- 程序计数器
- 寄存器
- 共享内存

SM处理的每个线程束计算所需资源属于片上（on-chip）资源，切换执行上下文（线程束的切换）没有时间损害
> 寄存器和共享内存都是片上资源
> 都有32位的寄存器组

对于给定内核，同时存在于同一个SM的线程块和线程束数量取决于SM中可用内核所需寄存器和共享内存的数量的分配 
 
每个线程消耗的寄存器越多，可以放在一个SM的线程束就越少
如果减少内核消耗寄存器的数量，SM可以同时处理更多线程束
> 前情提要：每个SM可以被分配多个线程块（SM的线程束是并发执行，线程束里面的线程是并行执行）



![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424160149681.png)


一个线程块消耗的共享内存越多，一个SM中可以同时处理的线程块越少
如果每个线程块使用的共享内存数量变少，就可以同时处理更多的线程块
> 要尽量增加线程块数量，提高并行（这里是并发？）

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424161107881.png)


## SM 占用率

计算资源被分配给线程块，线程块被称为活跃的块，线程块包含的线程束被称为活跃的线程束：
- 选定的线程束（正在执行）
- 阻塞的线程束（没做好准备执行）
- 符合条件的线程束（还没执行）
执行条件：
- 32个cuda核心可用于执行
- 指令的所有参数准备就绪

占用率（每个SM）：活跃线程束数量/最大线程束数量
                      1546/32 = 48

计算能力8.9为例
1. 一个SM最多线程块个数为Nb=24
2. 一个SM最多拥有的线程个数为Nt=1536
> 系统指定的，寄存器数量为64k SM数量为48 SM共享内存上限100K 单个线程共享内存上限99K

并行规模足够大的前提（核函数定义的总线程足够多）分析SM占用率：
1. 每个线程寄存器和共享内存使用少，线程块不小于64（Nt/Nb）时，100%
> 这里指的是每个线程块中的线程数 1536 / 24 = 64

2. 有限寄存器，当SM最多可驻留1536个，核函数每个线程最多使用42个寄存器
> `42(每个线程使用的寄存器数量) *1536 ==  64 * 1024 * 1024`

3. 有限空闲内存，若线程块大小定义为64，每个SM需要激活24个线程块才拥有1536个线程，达到100%利用率每个线程块可分配4.16KB的共享内存 （100 / 24）
一个线程块需要的共享内存超过99KB，核函数无法确定

网格和线程块大小的准则：
1. 线程块线程数量是线程束大小的整数倍
2. 线程块不要太小，根据内核资源调整线程块的大小
3. 线程块数量要远大于SM数量，保证设备有足够并行

# 延迟隐藏

# 避免线程束分化

避免线程束中的32个线程执行不同的指令

## 什么是线程束分化

一个线程束中的线程执行不同分支的指令，就会造成线程束分支
```
if (tid % 2 == 0) { a = 10.0f;}
else {b = 20.0f;}
```

紫色表示在等待，因为一个线程束是同时执行32个线程的  
![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424105148164.png)
- 线程束分化会降低并行能力，分支越多越严重
- 线程束分支只发生在同一个线程束中，不同线程束不会发生线程束分化
- 为获取最佳性能，避免同一个线程束中有不同的执行路径

常见解决思路：
- 不同线程束执行不同指令
```
if (（tid / 32） % 2 == 0) { a = 10.0f;}
else {b = 20.0f;}
```


## 并行规约计算

向量中满足交换律和结合律的运算成为规约问题，并行执行的称为并行规约计算

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424105551403.png)

![](https://picture-zikun.oss-cn-nanjing.aliyuncs.com/typora_PC/20250424172154599.png)

512个线程，每个线程束有32个线程，512 / 32 得到16个线程束

第一个：
- 严重的线程分化
- 第一轮，16个线程束都参与，但是每个线程束只有一半（16个线程）的参与计算

第二个
- 前8个进行规约，后面8个什么都不做。。。
- 最后五轮（数据只有32 16 8 4 2的时候，需要16 8 4 2 1和线程来进行规约计算），数据数量少于线程束大小，也会产生线程束分化（必须要使用一个线程束）（无法避免）