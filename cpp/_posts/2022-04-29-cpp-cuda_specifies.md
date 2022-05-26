---
layout: post
title: cuda cpp specifiers
sitemap: false
---

**참고**  
[1] <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers>  
* * *  

* toc
{:toc}

## __global__ function
* \__global__ void func(...)
* **device 상**에서 실행된다.
* **host에서 부를 수 있다.**
* **void return type** 만을 가질 수 있다.
* **class의 멤버로 사용할 수 없다.**
* device상에서 결과가 완료하기 전에 return을 먼저하는 asynchronous 이다.

## __device__ function
* \__device__ type func(...)
* device 상에서 실행된다.
* **host에서 부를 수 없다.** device 상에서 실행되는 함수에서만 부를 수 있다.
* \__global__ 과 함께 사용될 수 없다.

## __host__ function
* \__host__ type func(...)
* **host 상에서 실행된다.**
* **device에서 부를 수 없다.**
* \__global__ 과 함께 사용될 수 없다. \__device__ 와 함께 사용될 수 있다. \__host__ \__device__ type func(...)

## __device__ variable
* \__device__ type variable
* **global memory space**에 거주한다.
* **Cuda context와 lifetime을 같이 한다.**
* device 마다 구분되는 object를 가지고 있다. (무슨 말이지)
* **모든 threads로 부터 접근할 수 있다.**

## __constant__ variable
* \__constant__ type variable
* **constant memory space**에 거주한다.
* Cuda context와 lifetime을 같이 한다.
* device 마다 구분되는 object를 가지고 있다.
* 모든 threads로 부터 접근할 수 있다.

## __shared__ variable
* \__shared__ type variable
* **thread block의 shared memory space**에 거주한다.
* **block의 life time**을 같이 한다.
* block 마다 구분되는 object를 가지고 있다.
* **block 안의 thread로 부터 접근할 수 있다.**
* constant address를 가지고 있지 않다.