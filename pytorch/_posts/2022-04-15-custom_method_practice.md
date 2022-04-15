---
layout: post
title: Custom method practice
sitemap: false
---

**참고**  
[1] <https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html>  
* * *  

* toc
{:toc}

## custom method practice
> * custom method 예시 중 하나로 convolution layer와 batch norm layer를 **하나의 layer로** 합치는 실습을 해본다.
> * convolution과 batch norm은 forward 중에 backward 미분 계산을 위해 **input 텐서와 weight 텐서를 저장하면서 forward를 진행한다.** 이것은 layer가 깊어질 수록, 넓어질 수록 메모리를 많이 잡아먹는다.
> * 두 layer가 연속적으로 온다는 것을 가정하여 입력 텐서를 **한번만** 저장하면 되므로 메모리를 많이 지킬 수 있다.

## fusing convolution and batch norm
> * convolution 계산을 먼저한 후에 batch norm 계산을 한다.
> * 계산의 편리성을 위해 convolution의 파라미터로 bias=False, stride=1, padding=0, dilation=1, and groups=1 로 제한한다.
> * 마찬가지로 batch norm의 파라미터로 eps=1e-3, momentum=0.1, affine=False, and track_running_statistics=False 으로 제한한다.

## convolution backward function
> * convolution의 backward를 정의한다. 
> * convolution의 backward 함수는 [요기](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)를 보자.
~~~py
def convolution_backward(grad_out, X, weight):
    grad_w = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_w
~~~

## batch norm backward function
> * batch norm의 backward를 정의한다.
> * batch norm의 backward 함수는 [요기](http://kevinzakka.github.io/2016/09/14/batch_normalization/)를 보자.
~~~py
def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: out = (X - mean(X)) / (sqrt(var(X)) + eps)
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps)**2  # d_denom = -num / denom**2
    d_var = d_denom / (2 * sqrt_var)  # denom = torch.sqrt(var) + eps
    # Compute d_mean_dx before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # d_mean_dx has already been reassigned to a C-sized buffer so no need to worry

    # (1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # sqrt_var + eps > 0!
    return grad_input
~~~

## Fusing convolution and batch norm
> * convolution과 batch norm을 연속적으로 실행하는 새로운 Function을 만든다.
> * backward를 위해 한번의 인풋 데이터만 저장한다.
~~~py
class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        # Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight)
        # Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, conv_weight, = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight)
        return grad_X, grad_input, None, None, None, None, None
~~~

## modulize
~~~py
import torch.nn as nn
import math

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
~~~