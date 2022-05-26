---
layout: post
title: device memory
sitemap: false
---

**참고**  
[1] <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory>  
* * *  

* toc
{:toc}

## Cuda Runtime Initialization
* Cuda runtime을 위한 명확한 initialization function은 존재하지 않는다. 
* 모든 코드는 host에서 실행되며, 어떤 runtime function이 실행되면 그때서야 Cuda가 runtime에 들어선다.
* 첫번째 runtime function이 실행되면 **시스템** 각각의 **device에 primary context가 생성되며**, 함수가 실행된다.
* device의 primary context는 [driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interoperability-between-runtime-and-driver-apis)를 통해서 접근할 수 있다.
* host에서 $$ cudaDeviceReset() $$ 함수를 콜하면 해당 device의 primary context를 없앨 수 있다.


## Device memory
* Cuda 프로그래밍 모델은 시스템이 각각의 memory를 가진 host와 device로 이루어져 있다.
* Cuda runtime은 **host와 device 각각의 메모리**에 데이터를 **할당, 해제, 복사, 전송** 등의 처리를 할 수 있다.
* Device memory는 **linear memory**나 **CUDA arrays**를 통해서 할당될 수 있다. 일반적으로 사용되는 것은 **linear memory**이며, CUDA arrays는 Texture, Surface 메모리 할당시 필요로 한다.
* Linear memory는 일반적으로 사용하는 **single unified address space 주소체계이다.** 
* Linear memory는 일반적으로 **$$ cudaMalloc() $$ 함수**를 통해서 할당되고, **$$ cudaFree() $$ 함수를 통해서 해제**되고, **$$ cudaMemcpy() $$ 함수**를 통해서 host memory와 device memory 간의 **데이터 전송**이 이루어진다.

~~~cpp
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = 2048;
    size_t size = N * sizeof(float); // size_t는 해당 머신에서 가장 큰 unsigned 정수형 데이터형을 나타낸다.

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i=0;i<N;i++){
        h_A[i] = i;
        h_B[i] = 2*i;
        h_C[i] = 0;
    }

    // Allocate vectors in device memory
    // cuda memory를 사용함에도 float* 변수를 사용했다는 것.
    // 변수가 cuda memory를 참조하고 있다는 것을 알려야 하기 때문에 포인터의 참조형을 전달한다.
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // To, From, size, ...
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); // To, From, size, ...

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 총 N개 이상의 thread가 필요하도록
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // runtime function

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}
~~~
* host와 device간에 **global 변수**에 데이터 **전송**하는 방법은 아래와 같다.
* \__constant__, \__device__ 와 Symbol 메소드를 사용하면 된다.
~~~cpp
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data)); // to device
cudaMemcpyFromSymbol(data, constData, sizeof(data)); // from device

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
~~~