---
layout: post
title: Custom method
sitemap: false
---

**참고**  
[1] <https://pytorch.org/docs/stable/notes/extending.html#extending-autograd>  
* * *  

* toc
{:toc}

## custom operation
> * 파이토치를 사용하다보면 기존에 라이브러리에 있는 op 외에 다른 op를 사용하고 싶으면 미분 가능한, Backpropa가 가능한 op를 만들어야 한다.
> * torch.autograd.Function 클래스를 이용하여 새로운 op를 만들 수 있다.
> * 미분 가능한 새로운 op를 만든 후, 자체적인 weight 변수들을 가진 모듈을 만들고 싶다면 torch.nn.module로 감쌓으면 된다.
> * op를 만들기 위해서는 네가지 step이 존재한다.
> 
> ### step1
> > * torch.autograd.Function의 subclass로 class를 만든 후에 forward static method와 backward static method를 정의해야한다.
> > * forward() 메소드는 원하는 만큼 인자를 받을 수 있다. 설령 **tensor가 아니더라도** 인자로 받을 수 있다.
> > * forward() 메소드는 **single tensor나 tuple of tensors** 를 출력해야한다.
> > * backward() 메소드는 gradient 계산을 위한 메소드이다. forward() 와는 달리 **인자의 형태가 정해져 있다.** (ctx, grad_output)
> > * backward() 메소드는 **output 또한 규칙이 정해져 있으며**, input으로 받은 tensor의 gradient를 **그대로 output으로 출력해야한다.** 입력 파라미터가 **tensor가 아니라면 None**을 출력하면 된다.
> 
> ### step2
> > * 새로 만든 op가 올바르게 돌아가기 위해서는 forward() 메소드 안에 있는 **ctx 객체**를 잘 사용해야 한다.
> > * ctx는 forward 시에 그래프 형태로 저장되어 있다가 backward()의 인자로 들어간다.
> > * ctx 객체는 다음과 같은 메소드들을 가지고 있다.
> > * save for backward() : 미분을 계산하기 위해 forward() 수행 시에 저장해야 하는 데이터이다.
> > * mark_dirty() : forward() 메소드 실행 시, 입력 텐서가 수정되서는 안되는데, 수정될 경우 mark_dirty(modified tensor)를 통해 알려야만 한다.
> > * mark_non_diffrentiable() : output tensor가 미분이 불가능하는 경우 알려줘야 한다. mark_non_diffrentiable(output tensor)
> > * set_materialize_grads() : 만약 미분을 계산하는데 input이 필요 없다면, set_materialize_grads(False) 설정을 통해 계산 최적화를 할 수도 있다. default는 True이다.
> 
> ### step3 
> > * 만약에 우리가 만든 op가 double backward를 지원하지 않는다면 **once_differentiable()** 를 통해 한번만 미분 가능하다는 것을 명시해야한다.
> 
> ### step4
> > * torch.autograd.gradcheck()을 통해 올바르게 gradient가 계산되는지 확인하자.

## Example
~~~py
class LinearFunction(Function):
    
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        # gradient를 계산하기 위해서는 input, weight, bias 텐서가 필요하다.
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None # output으로 나가야하는 gradient, forward의 input과 동일한 구조여야만 한다.

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
~~~
~~~py
linear = LinearFunction.apply
~~~

## modulize
> * autograd.Function을 이용하여 custom operation을 만들었다면, 그것을 이용해서 **파라미터와 버퍼를 저장할 수 있는** 모듈 torch.nn.Module 을 만들 수 있다.
> * 새로운 custom module을 만들기 위해서는 torch.nn.Module 클래스를 상속하고 **\__init__ 메소드와 forward 메소드**를 정의해야한다.
> * **backward 메소드는 필요하지 않다.** forward 에서 Function.apply을 호출하기 때문에 autograd 엔진이 자동으로 backward 함수를 내장하여 그래프를 그린다.
> * \__init__ : Module을 정의하는 역할.
> * forward : 주로 Function.apply()를 그대로 실행한다.
> ~~~py
> class Linear(nn.Module):
>     def __init__(self, input_features, output_features, bias=True):
>         super(Linear, self).__init__()
>         # 일반 attribute는 모델 저장 시에 저장되지 않는다. Parameter 형태만이 모델 저장에 저장되는 attribute이다.
>         self.input_features = input_features
>         self.output_features = output_features
> 
>         # nn.Parameter is a special kind of Tensor, that will get
>         # automatically registered as Module's parameter once it's assigned
>         # as an attribute. Parameters and buffers need to be registered, or
>         # they won't appear in .parameters() (doesn't apply to buffers), and
>         # won't be converted when e.g. .cuda() is called. You can use
>         # .register_buffer() to register buffers.
>         # nn.Parameters require gradients by default.
>         self.weight = nn.Parameter(torch.empty(output_features, input_features))
>         if bias:
>             self.bias = nn.Parameter(torch.empty(output_features))
>         else:
>             # You should always register all possible parameters, but the
>             # optional ones can be None if you want.
>             self.register_parameter('bias', None)
> 
>         # Not a very smart way to initialize weights
>         nn.init.uniform_(self.weight, -0.1, 0.1)
>         if self.bias is not None:
>             nn.init.uniform_(self.bias, -0.1, 0.1)
> 
>     def forward(self, input):
>         # See the autograd section for explanation of what happens here.
>         return LinearFunction.apply(input, self.weight, self.bias)
> 
>     def extra_repr(self):
>         # (Optional)Set the extra information about this module. You can test
>         # it by printing an object of this class.
>         return 'input_features={}, output_features={}, bias={}'.format(
>             self.input_features, self.output_features, self.bias is not None
>         )
> ~~~