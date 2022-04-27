---
layout: post
title: custom cuda extension to pytorch
sitemap: false
---

**참고**  
[1] <https://pytorch.org/tutorials/advanced/cpp_extension.html>  
* * *  

* toc
{:toc}

## LLTM in python
> * LSTM을 변형한 새로운 operation LLTM을 python으로 한번 작성해봅니다.
> ~~~py
> class LLTM(torch.nn.Module):
>     def __init__(self, input_features, state_size):
>         super(LLTM, self).__init__()
>         self.input_features = input_features
>         self.state_size = state_size
>         # 3 * state_size for input gate, output gate and candidate cell gate.
>         # input_features + state_size because we will multiply with [input, h].
>         self.weights = torch.nn.Parameter(
>             torch.empty(3 * state_size, input_features + state_size))
>         self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
>         self.reset_parameters()
> 
>     def reset_parameters(self):
>         stdv = 1.0 / math.sqrt(self.state_size)
>         for weight in self.parameters():
>             weight.data.uniform_(-stdv, +stdv)
> 
>     def forward(self, input, state):
>         old_h, old_cell = state
>         X = torch.cat([old_h, input], dim=1)
> 
>         # Compute the input, output and candidate cell gates with one MM.
>         gate_weights = F.linear(X, self.weights, self.bias)
>         # Split the combined gate weight matrix into its components.
>         gates = gate_weights.chunk(3, dim=1)
> 
>         input_gate = torch.sigmoid(gates[0])
>         output_gate = torch.sigmoid(gates[1])
>         # Here we use an ELU instead of the usual tanh.
>         candidate_cell = F.elu(gates[2])
> 
>         # Compute the new cell state.
>         new_cell = old_cell + candidate_cell * input_gate
>         # Compute the new hidden state and output.
>         new_h = torch.tanh(new_cell) * output_gate
> 
>         return new_h, new_cell
> ~~~

## Building C++
> * C++ 코드를 python에서 사용하기 위해서는 빌드를 해야만 한다.
> * **setuptools**를 이용해서 미리 빌드한 실행 파일을 사용할 수 있고, **jit**를 이용해서 실시간으로 빌드해서 사용할 수도 있다.
> * 아래의 코드는 setuptools를 이용해서 C++ 파일을 빌드한 것이다.
> * 아래와 같이 setup.py 코드를 작성해서 python setup.py install 명령어를 실행하면 **pip 라이브러리에 lltm_cpp**가 추가된다.
> * 사용할 때는 **torch를 꼭 먼저 import하고** 사용해야만 한다.  
> ~~~py
> from setuptools import setup, Extension
> from torch.utils import cpp_extension
> setup(name='lltm_cpp',
>       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
>       cmdclass={'build_ext': cpp_extension.BuildExtension})
> ~~~
> * 아래와 같이 실행하고 싶은 python 파일에 작성하면 실시간으로 빌드되어 사용할 수 있다.
> ~~~py
> from torch.utils.cpp_extension import load
> lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
> ~~~

## LLTM in C++
> * C++ 코드에서 torch의 여러 자료형이나 함수들을 사용하기 위해서는 **torch/extension.h** 헤더를 import 한다.
> * torch/extension.h 헤더 파일은 ATen 라이브러리를 포함하고, 이것은 **텐서의 계산을 위한 주요 API들이** 내장되어 있다.
> * torch/extension.h 헤더 파일은 C++ 코드를 파이썬으로 바인딩하기 위한 코드들이 내장되어 있다.
> * ATen 관련 API들은 [여기](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)있다.
> * 아래의 코드는 LLTM의 forward와 backward를 C++ 언어로 작성한 뒤, **pybind11**을 이용해서 C++ 함수나 클래스를 **파이썬으로 바인딩** 시키는 코드이다.
> * 아래의 코드를 작성한 이후 python setup.py install 명령어로 빌드한 후 사용할 수 있다.  
> 
> ~~~cpp
> #include <torch/extension.h>
> #include <iostream>
> #include <vector>
> 
> torch::Tensor d_sigmoid(torch::Tensor z) {
>   auto s = torch::sigmoid(z);
>   return (1 - s) * s;
> }
> 
> // tanh'(z) = 1 - tanh^2(z)
> torch::Tensor d_tanh(torch::Tensor z) {
>   return 1 - z.tanh().pow(2);
> }
> 
> // elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
> torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
>   auto e = z.exp();
>   auto mask = (alpha * (e - 1)) < 0;
>   return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
> }
> 
> std::vector<at::Tensor> lltm_forward(
>     torch::Tensor input,
>     torch::Tensor weights,
>     torch::Tensor bias,
>     torch::Tensor old_h,
>     torch::Tensor old_cell) {
>   auto X = torch::cat({old_h, input}, /*dim=*/1);
> 
>  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
>  auto gates = gate_weights.chunk(3, /*dim=*/1);
>
>  auto input_gate = torch::sigmoid(gates[0]);
>  auto output_gate = torch::sigmoid(gates[1]);
>  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);
> 
>   auto new_cell = old_cell + candidate_cell * input_gate;
>   auto new_h = torch::tanh(new_cell) * output_gate;
> 
>   return {new_h,
>           new_cell,
>           input_gate,
>           output_gate,
>           candidate_cell,
>           X,
>           gate_weights};
> }
> 
> std::vector<torch::Tensor> lltm_backward(
>     torch::Tensor grad_h,
>     torch::Tensor grad_cell,
>     torch::Tensor new_cell,
>     torch::Tensor input_gate,
>     torch::Tensor output_gate,
>     torch::Tensor candidate_cell,
>     torch::Tensor X,
>     torch::Tensor gate_weights,
>     torch::Tensor weights) {
>   auto d_output_gate = torch::tanh(new_cell) * grad_h;
>   auto d_tanh_new_cell = output_gate * grad_h;
>   auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;
> 
>   auto d_old_cell = d_new_cell;
>   auto d_candidate_cell = input_gate * d_new_cell;
>   auto d_input_gate = candidate_cell * d_new_cell;
> 
>   auto gates = gate_weights.chunk(3, /*dim=*/1);
>   d_input_gate *= d_sigmoid(gates[0]);
>   d_output_gate *= d_sigmoid(gates[1]);
>   d_candidate_cell *= d_elu(gates[2]);
> 
>   auto d_gates =
>       torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);
> 
>   auto d_weights = d_gates.t().mm(X);
>   auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);
> 
>   auto d_X = d_gates.mm(weights);
>   const auto state_size = grad_h.size(1);
>   auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
>   auto d_input = d_X.slice(/*dim=*/1, state_size);
> 
>   return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
> }
> 
> PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
>   m.def("forward", &lltm_forward, "LLTM forward");
>   m.def("backward", &lltm_backward, "LLTM backward");
> }
> ~~~
> * 빌드가 완료된 이후에는 python 라이브러리에 포함되어 다른 주소에서도 lltm_cpp 패키지를 임포트할 수 있다.
> * 아래의 코드는 빌드된 패키지를 임포트하여 파이썬 모듈을 만드는 코드이다.  
> 
> ~~~py
> import math
> import torch
> 
> # Our module!
> import lltm_cpp
> 
> class LLTMFunction(torch.autograd.Function):
>     @staticmethod
>     def forward(ctx, input, weights, bias, old_h, old_cell):
>         outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
>         new_h, new_cell = outputs[:2]
>         variables = outputs[1:] + [weights]
>         ctx.save_for_backward(*variables)
> 
>         return new_h, new_cell
> 
>     @staticmethod
>     def backward(ctx, grad_h, grad_cell):
>         outputs = lltm_cpp.backward(
>             grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
>         d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
>         return d_input, d_weights, d_bias, d_old_h, d_old_cell
> 
> 
> class LLTM(torch.nn.Module):
>     def __init__(self, input_features, state_size):
>         super(LLTM, self).__init__()
>         self.input_features = input_features
>         self.state_size = state_size
>         self.weights = torch.nn.Parameter(
>             torch.empty(3 * state_size, input_features + state_size))
>         self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
>         self.reset_parameters()
> 
>     def reset_parameters(self):
>         stdv = 1.0 / math.sqrt(self.state_size)
>         for weight in self.parameters():
>             weight.data.uniform_(-stdv, +stdv)
> 
>     def forward(self, input, state):
>         return LLTMFunction.apply(input, self.weights, self.bias, *state)
> ~~~
> * 위의 코드는 순수 파이썬으로 짠 코드에 비해 forward pass 에서는 높은 성능 향상을 보였지만, backward pass 에서는 큰 차이가 없었다. 이것은 pytorch의 backward 처리 과정(autograd)은 최적화가 그 만큼 많이 들어가 있다는 것을 보여준다.
> * pytorch의 ATen 백엔드는 위와 같은 유저가 빌드한 코드를 **cpu와 gpu**에서 돌아갈 수 있도록 해준다. 위의 코드는 cpu 메모리의 데이터를 입력으로 받으면 cpu에서 처리하고, gpu 메모리에 있는 데이터를 받으면 gpu에서 자동으로 병렬처리하여 계산한다.

## LLTM in Cuda
> * C++로만 작성한 코드는 파이썬으로만 작성한 코드에 비해 월등한 성능 향상을 보여줬지만, kernel 함수를 직접 작성하면 여기서 더 성능 향상을 할 수 있다.
> * LLTM은 중간에 gate를 **세개의 부분으로 나눠서 각각 계산을 한다**. 이것은 각각 **독립의 관계를 가지기에 병렬처리할 수도 있다**.
> * 위의 코드를 병렬처리 하기 위해서는 새로운 함수를 작성해야 하고 병렬처리 해줘야 한다.
> * 이를 위해 새로운 kernel function을 만들어야 한다.
> * 대개의 경우 python에서 바인딩할 수 있는 C++ 파일을 우선 작성한다.
> * 해당 C++ 파일과 Cuda 파일을 빌드할 시, cpp_extension은 C++ 코드는 gcc 컴파일러를 이용하고, Cuda 코드는 nvcc 컴파일을 이용해서 컴파일 후 합친다.
> 
> ~~~cpp
> // lltm_cuda.cpp
> #include <torch/extension.h>
> 
> #include <vector>
> 
> // CUDA forward declarations
> 
> // cuda function... .cu 파일에 동일한 이름의 함수가 있어야 한다.
> std::vector<torch::Tensor> lltm_cuda_forward(
>     torch::Tensor input,
>     torch::Tensor weights,
>     torch::Tensor bias,
>     torch::Tensor old_h,
>     torch::Tensor old_cell);
> 
> // cuda function... .cu 파일에 동일한 이름의 함수가 있어야 한다.
> std::vector<torch::Tensor> lltm_cuda_backward(
>     torch::Tensor grad_h,
>     torch::Tensor grad_cell,
>     torch::Tensor new_cell,
>     torch::Tensor input_gate,
>     torch::Tensor output_gate,
>     torch::Tensor candidate_cell,
>     torch::Tensor X,
>     torch::Tensor gate_weights,
>     torch::Tensor weights);
> 
> // C++ interface
> 
> #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
> #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
> #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
> 
> // python에서 바인딩할 c++ function
> std::vector<torch::Tensor> lltm_forward(
>     torch::Tensor input,
>     torch::Tensor weights,
>     torch::Tensor bias,
>     torch::Tensor old_h,
>     torch::Tensor old_cell) {
>   CHECK_INPUT(input);
>   CHECK_INPUT(weights);
>   CHECK_INPUT(bias);
>   CHECK_INPUT(old_h);
>   CHECK_INPUT(old_cell);
> 
>   return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
> }
> 
> // python에서 바인딩할 c++ function
> std::vector<torch::Tensor> lltm_backward(
>     torch::Tensor grad_h,
>     torch::Tensor grad_cell,
>     torch::Tensor new_cell,
>     torch::Tensor input_gate,
>     torch::Tensor output_gate,
>     torch::Tensor candidate_cell,
>     torch::Tensor X,
>     torch::Tensor gate_weights,
>     torch::Tensor weights) {
>   CHECK_INPUT(grad_h);
>   CHECK_INPUT(grad_cell);
>   CHECK_INPUT(input_gate);
>   CHECK_INPUT(output_gate);
>   CHECK_INPUT(candidate_cell);
>   CHECK_INPUT(X);
>   CHECK_INPUT(gate_weights);
>   CHECK_INPUT(weights);
> 
>   return lltm_cuda_backward(
>       grad_h,
>       grad_cell,
>       new_cell,
>       input_gate,
>       output_gate,
>       candidate_cell,
>       X,
>       gate_weights,
>       weights);
> }
> 
> PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
>   m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
>   m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
> }
> ~~~
> * nvcc는 C++의 여러 표준 라이브러리를 컴파일할 수 있고, 또한 torch의 여러 api 또한 컴파일할 수 있기 때문에, **cuda 파일에서도 torch의 여러 api들을 사용할 수 있고** 이것은 매우 최적화가 잘되어있는 api들이라 사용하는 것을 권한다.
> * 다만 **C++ 파일과 Cuda 파일의 이름은 달라야만 한다**.
> * GPU 연산과 텐서는 static이 아닌 run time에서 타입을 결정하여 사용할 수 있다. 때문에 template 구문을 사용하여 run time에 타입을 결정할 수 있다.
> * AT_DISPATCH_FLOATING_TYPES은 GPU 연산을 실행할 시에 연산 데이터의 타입을 다룬다. 함수로 받는 인자를 floating, double 형만을 받겟다는 의미이고, 첫번째 인자로 타입 형을 넣어줘야한다.
> * AT_DISPATCH_FLOATING_TYPES에 들어가는 lambda function을 보면 scalar_t를 받는 것을 알 수 있는데, 이것은 run time에 타입을 결정할 수 있는 template 구문이랑 동일하다.
> * AT_DISPATCH_ALL_TYPES은 모든 타입을 받을 수 있는 구문이다.
> * 잘은 모르겠지만 kernel에 들어가는 데이터형은 index가 어떻게 구성되는지를 모르겟다. 여기서는 들어가는 데이터는 2차원의 모양인데, 1차원의 데이터로 vectorize 된다... 일단은 pytorch의 텐서 데이터는 무조건 다 펴진다고 생각하자. 아니면 2차원 데이터임에도 1차원으로 접근할 수 있는 건가?
> 
> ~~~c++
> #include <torch/extension.h>
> 
> #include <cuda.h>
> #include <cuda_runtime.h>
> 
> #include <vector>
> 
> template <typename scalar_t>
> __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
>   return 1.0 / (1.0 + exp(-z));
> }
> 
> template <typename scalar_t>
> __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
>   const auto s = sigmoid(z);
>   return (1.0 - s) * s;
> }
> 
> template <typename scalar_t>
> __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
>   const auto t = tanh(z);
>   return 1 - (t * t);
> }
> 
> template <typename scalar_t>
> __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
>   return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
> }
> 
> template <typename scalar_t>
> __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
>   const auto e = exp(z);
>   const auto d_relu = z < 0.0 ? 0.0 : 1.0;
>   return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
> }
> 
> std::vector<torch::Tensor> lltm_cuda_forward(
>     torch::Tensor input,
>     torch::Tensor weights,
>     torch::Tensor bias,
>     torch::Tensor old_h,
>     torch::Tensor old_cell) {
>   auto X = torch::cat({old_h, input}, /*dim=*/1); // torch의 api를 그대로 사용할 수 있는 것을 볼 수 있다.
>   auto gates = torch::addmm(bias, X, weights.transpose(0, 1));
> 
>   const auto batch_size = old_cell.size(0);
>   const auto state_size = old_cell.size(1);
> 
>   auto new_h = torch::zeros_like(old_cell);
>   auto new_cell = torch::zeros_like(old_cell);
>   auto input_gate = torch::zeros_like(old_cell);
>   auto output_gate = torch::zeros_like(old_cell);
>   auto candidate_cell = torch::zeros_like(old_cell);
> 
>   // 계산은 point-wise로 이루어진다. 즉 데이터의 shape만큼 thread가 존재하면 된다.
>   // new_cell의 shape이 [b, state_size] 이므로 b * state_size / threads 만큼의 블럭이 존재하면 되는데, 정확하게 나누어 떨어지지 않을 경우, 하나의 block이 더 추가되어야한다.(올림)
>   const int threads = 1024;
>   const dim3 blocks((state_size + threads - 1) / threads, batch_size);
> 
>   // lltm_cuda_forward_kernel 함수를 실행한다. runing time에 타입이 결정되는 scalar_t 타입형을 사용할 수 있다.
>   AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
>     lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
>         gates.data<scalar_t>(),
>         old_cell.data<scalar_t>(),
>         new_h.data<scalar_t>(),
>         new_cell.data<scalar_t>(),
>         input_gate.data<scalar_t>(),
>         output_gate.data<scalar_t>(),
>         candidate_cell.data<scalar_t>(),
>         state_size);
>   }));
> 
>   return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
> }
> 
> template <typename scalar_t>
> __global__ void lltm_cuda_forward_kernel(
>     const scalar_t* __restrict__ gates,
>     const scalar_t* __restrict__ old_cell,
>     scalar_t* __restrict__ new_h,
>     scalar_t* __restrict__ new_cell,
>     scalar_t* __restrict__ input_gate,
>     scalar_t* __restrict__ output_gate,
>     scalar_t* __restrict__ candidate_cell,
>     size_t state_size) {
> 
>   // gates : batchsize * (3 * state_size)
>   // gate : batchsize * state_size
>   // block : (~, batchsize)
>   // blockDim.x = 1024
>   // column이 state_size 보다 클 수도 있음. 정확하게 안 나눠지는 경우...
>   const int column = blockIdx.x * blockDim.x + threadIdx.x;
>   const int index = blockIdx.y * state_size + column;
>   const int gates_row = blockIdx.y * (state_size * 3);
>   if (column < state_size) {
>     input_gate[index] = sigmoid(gates[gates_row + column]);
>     output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
>     candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
>     new_cell[index] =
>         old_cell[index] + candidate_cell[index] * input_gate[index];
>     new_h[index] = tanh(new_cell[index]) * output_gate[index];
>   }
> }
> ~~~

## Using accessors
> * 위와 같이 커널 함수안에서 포인터에 직접 접근하는 것은 텐서의 차원이 올라갈수록 계산하기 어려워지고 보기 좋지 않게 된다.
> * accessors 메소드를 사용하면 직접 계산하지 않아도 텐서의 타입과 차원을 dynamic 하게 체크할 수 있다. **정확한 shape을 입력하지 않아도** index 만을 보고 자동으로 나눠서 계산해주나 보다.
> * 계산이 자동으로 된다는 편리성이 있지만, 정확하지 않을수도 있을 것 같아서 사용해 주의를 하자.
> * accessors 메소드를 사용하면 싱글 pointer로 전환하지 않아도 된다.
> * cpu 상에서 계산하기 위해서는 **.accessor\<type, dimension>** 선언하고, cuda 상에서 계산하기 위해서는 **.packed_accessor32\<type, dimension>** 으로 선언한다.  
> ~~~c++
> template <typename scalar_t>
> __global__ void lltm_cuda_forward_kernel(
>     const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> gates,
>     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> old_cell,
>     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_h,
>     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_cell,
>     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_gate,
>     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_gate,
>     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> candidate_cell) {
>   //batch index
>   const int n = blockIdx.y;
>   // column index
>   const int c = blockIdx.x * blockDim.x + threadIdx.x;
>   if (c < gates.size(2)){
>     input_gate[n][c] = sigmoid(gates[n][0][c]);
>     output_gate[n][c] = sigmoid(gates[n][1][c]);
>     candidate_cell[n][c] = elu(gates[n][2][c]);
>     new_cell[n][c] =
>         old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
>     new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
>   }
> }
> ~~~
> ~~~c++
> std::vector<torch::Tensor> lltm_cuda_forward(
>     torch::Tensor input,
>     torch::Tensor weights,
>     torch::Tensor bias,
>     torch::Tensor old_h,
>     torch::Tensor old_cell) {
>   auto X = torch::cat({old_h, input}, /*dim=*/1);
>   auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
> 
>   const auto batch_size = old_cell.size(0);
>   const auto state_size = old_cell.size(1);
> 
>   auto gates = gate_weights.reshape({batch_size, 3, state_size});
>   auto new_h = torch::zeros_like(old_cell);
>   auto new_cell = torch::zeros_like(old_cell);
>   auto input_gate = torch::zeros_like(old_cell);
>   auto output_gate = torch::zeros_like(old_cell);
>   auto candidate_cell = torch::zeros_like(old_cell);
> 
>   const int threads = 1024;
>   const dim3 blocks((state_size + threads - 1) / threads, batch_size);
> 
>   AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
>     lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
>         gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
>         old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
>         new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
>         new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
>         input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
>         output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
>         candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
>   }));
> 
>   return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
> }
> ~~~