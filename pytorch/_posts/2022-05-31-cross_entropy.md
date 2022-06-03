---
layout: post
title: Cross Entropy
sitemap: false
---

**참고**  
[1] <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>  
[2] <https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html>
* * *  

* toc
{:toc}

## Introduction
* 논문 코드를 구현하던 중 official 코드의 결과 값과 직접 구현한 코드의 결과 값의 차이가 많아 원인을 찾는 중에 발견한 그동안에 모르고 있었던 pytorch의 cross entropy 메소드에 대한 **치명적인 실수**에 대한 짧은 글이다.
* cross entropy는 **KL divergence**의 식에서 유래한 objective fucntion이다. 그렇기 때문에 함수의 입력으로 들어가는 텐서들은 **확률 값**을 나타내며 값이 0부터 1까지 **normalize** 되어 있어야 한다.
* 하지만 pytorch의 cross entropy method는 **unnormalized 데이터를 입력으로 받아** 내부에서 **softmax**를 통해 normalize를 시킨다.
* pytorch의 binary cross entropy method는 **normalized 데이터를 입력으로 받는다.**
* pytorch의 binary cross entropy with logits method는 **unnormlized 데이터를 입력으로 받아** 내부에서 sigmoid를 통해 normlized를 시킨다.

## Cross entropy
* Cross entropy는 **KL divergence**에서 유래한 식이다.
* $$ p(x) $$를 정답 labels라고 한다면 KL divergence 식에서 고정된 상수의 식을 제외한 항을 Cross Entropy라고 부르며, **KL divergence를 최소화하는 것은 Cross Entropy를 최소화하는 것과 같다.**
* 일반적으로 Classification 문제를 다룰 때, **objective function으로 cross entropy**를 많이 사용한다.

$$
\begin{aligned}
  D_{KL}(p||q) = E_{x \sim p}[log(p(x))] - E_{x \sim p}[log(q(x))] \\[1em]
  CrossEntropy : H(p, q) = - E_{x \sim p}[log(q(x))] \\[1em]
                          = - \sum p(x)log(q(x))
\end{aligned}
$$

## Binary Cross Entropy
* **경우의 수가 True, Falce 두개인** Binary Classification 문제를 풀 땐, objective function으로 **Binary cross entropy** 함수를 이용한다.
* 머신러닝 문제에서 입력되는 데이터를 $$ x $$라 칭하고 모델의 결과 값을 y라고 칭한다.
* **ground truth는 $$ p(y \| x) $$이며**, 확률 변수 y가 가질 수 있는 데이터가 0과 1이기 때문에 **$$ p(y \| x) $$ 는 Bernoulli distribution 이다.**
* **우리의 모델** 또한 likelihood function이 **Bernoulli distribution $$ q(y\|x) $$을 따른다고 가정하며**, **binary cross entropy 함수를 이용하여 bernoulli distribution의 parameter p = $$ q(y=1\|x) $$를 구하는 학습을 진행한다.**
* 대개의 경우 classification 문제이기 때문에 **ground truth $$p(y\|x)$$는 $$ p(y=1\|x) = 1 \text{ or } 0 $$ 의 값**을 나타낸다.
* $$P(y=1\|x) = 1$$일 경우

$$
\begin{aligned}
  CrossEntropy : H(p, q) = - E_{y|x \sim p}[log(q(y|x))] \\[1em]
                          = - \sum p(y|x)log(q(y|x)) \\[1em]
                          = -log(q(y=1|x)) \\[1em]
\end{aligned}
$$

* $$P(y=1\|x) = 0$$일 경우

$$
\begin{aligned}
  CrossEntropy : H(p, q) = - E_{y|x \sim p}[log(q(y|x))] \\[1em]
                          = - \sum p(y|x)log(q(y|x)) \\[1em]
                          = -(1-log(q(y=1|x))) \\[1em]
\end{aligned}
$$

* Pytorch에서 **binary cross entropy와 관련된 메소드는 두개**가 있다.
* torch.nn.functional.binary_cross_entropy() 함수는 input tensor와 target tensor가 **동일한 shape**을 가지고 있어야 하며, input tensor는 **nomalized tensor**여야만 한다.
* torch.nn.functional.binary_cross_entropy_with_logits() 함수도 마찬가지로 input tensor와 target tensor가 **동일한 shape**을 가지고 있어야 하며, input tensor는 **unnolized tensor**이고 내부에서 자체적으로 sigmoid 함수를 걸친다. 그렇기 때문에 따로 **모델의 결과 값에 sigmoid를 넣지 않도록 해야만 한다.**
* input tensor 와 target tensor가 동일한 shape을 가지고 있어야 하는 이유는 **tensor의 값 하나하나가 독립적인 하나의 확률 변수, 확률 분포라고 생각하기 때문이다.** 이것은 binary cross entropy 뿐만 아니라 cross entropy 함수도 동일하게 적용된다.

## cross entropy
* 주로 경우의 수가 **두개가 아닌 Classification 문제를 풀 때 사용하는 objective function이다.**
* 경우의 수가 두개가 아니기 때문에 ground truth는 Bernoulli distribution의 multinomial variable 버전?인 **Dirichlet distribution 이다.**
* ground truth는 $$ p(y\|x) $$이며, 확률 변수 **y = 0 ~ N 의 정수 값**을 가질 수 있다. ground truth는 one hot encoding하지 않도록 한다.
* 우리의 모델 또한 likelyhood function으로 Dirichlet distribution $$ q(y\|x) $$을 따르고, parameter가 **1개인 Bernoulli distribution과는 달리 여러개(N개)의 parameter**를 가지기 때문에 **결과 값도 N개가 되어야 한다.**
* binary variables 때와 마찬가지로 대개의 경우 classification 문제이기 때문에 ground truth는 **하나의 값에만 확률 값이 1**이고 나머지는 0인 분포를 띈다.
* $$ P(y=n\|x) = 1 $$ 일 경우

$$
\begin{aligned}
  CrossEntropy : H(p, q) = - E_{y|x \sim p}[log(q(y|x))] \\[1em]
                          = - \sum p(y|x)log(q(y|x)) \\[1em]
                          = -log(q(y=n|x))) \\[1em]
\end{aligned}
$$

* torch.nn.functional.cross_entropy() 함수는 input tensor와 target tensor가 **다른 shape을 띄고 있다.** input tensor는 **Dirichlet distribution의 파라미터 수(C)에 대해서의 값**을 가지고 있어야 하므로 **(C) 혹은 (N, C) 혹은 (N, C, d_1, d_2, ... , d_n)의 shape**을 가지고 **target tensor는 classification 문제이기 때문에 어떤 값이 1인지 만 나타내면 되어서 (,) 혹은 (N) 혹은 (N, d_1, d_2, ..., d_n)의 shape**을 가지고 있어야 한다. 혹은 **classification 문제가 아닌 일반적인 문제일 경우 target tensor는 input tensor와 동일한 shape을 가지고 있으면 된다.** 두 경우 모두 C를 제외하면 동일한 shape을 가지고 있어야 하며 **C 채널이 항상 가장 마지막에 있는 것이 아닌 두번째에 존재하는 것을 인지**해야 한다. 또한 **unnormalized input tensor**를 입력으로 받고 내부에서 **자체적으로 softmax 함수를 취한다.**
* bernoulli distribution의 경우에서는 입력 데이터 모두가 독립적인 확률 분포로 쓰였지만 **dirichlet distribution의 경우에서는 입력 텐서의 C채널이 모두 하나의 확률 분포**를 나타내기 위해 쓰여진다.

## Unnormalized input tensor를 입력으로 받는 이유
* 논리적으로는 사용자가 직접 sigmoid나 softmax 함수를 통해 normalize를 한 후에 cross entropy에 입력으로 넣는게 맞는 것 같지만, 최근 파이토치 함수에서는 Unnormalized input tensor을 입력으로 받고 자체적으로 normalize를 한다.
* 정확하게 그렇게 취한 이유는 잘 모르겠지만 사용자가 실수하는 것을 줄이기 위해 하는 건지... normalize 과정과 cross entropy 계산 과정을 합침으로써 더 학습에 안정적으로 수식을 수정하는 방법이 있어서 그런건지는 잘 모르겠다...
* 그냥 인지하고 있자.