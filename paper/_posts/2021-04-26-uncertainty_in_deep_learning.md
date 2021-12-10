---
layout: post
title: Uncertainty in Deep Learning -writting-
sitemap: false
---

**참고**  
[1] Yarin Gal, Uncertainty in Deep Learning, 2016  
* * *  

# 1. The Language of Uncertainty
## 1.1 Bayesian modelling
Regression, Classification 문제를 머신러닝으로 푼다고 가정해보자. 그러면 우리는 가장 먼저 머신러닝 모델을 정의하고 데이터들을 이용해서 손실 함수의 평균이 가장 낮도록 만드는 모델의 파라미터를 찾는 것으로 학습할 것이다. 하지만 이런식으로 학습이 된다면 현재 가지고 있는 데이터에만 최적화된 모델을 추구하고, 처음 접하는 데이터를 고려하지 않으면서 학습되기 때문에 일반화에 어려움이 있다. 또한 최적화된 파라미터를 구한다고 해도 그 파라미터가 얼마나 신뢰있는지 판단할 수 없다.
<br/>
이러한 문제를 해결할 수 있는 방법 중 하나로 **베이지안 접근**이 있다. 베이지안 접근은 문제를 풀기(여기서 문제를 푼다는 의미는 inference를 의미한다고 하자)에 앞서 모델 파라미터의 분포를 사전에 정의하고 진행한다. 이것을 "사전의 믿음" **사전분포 p(w)**라고 칭한다. 그 다음 데이터가 주어질 때 마다 이 사전분포는 해당 데이터에 적합하도록 수정 되고, 이렇게 수정된 분포를 **사후분포 p(w|D)**라 부른다. 좀 더 자세히 들어가면, 우선 어떤 식으로 문제를 풀던간에 **가장 먼저 해야할 것은 likelihood function을 정의**하는 것이다. **likelihood function이란 모델의 unknown parameter가 주어졌을때, 즉 파라미터에 의해 하나의 정해진 모델이 현재 주어진(sampling) 데이터에 얼마나 적합한지(fit)를 평가하는 요소이다.** supervised learning에서는 P(Y|X,w)를, unsupervised learning에서는 P(X|w)를 예로 들 수 있다.
<br/>
A Gaussian likelihood for regression:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/1.png"></p>
**model precision은 학습 데이터가 가진 노이즈에 의한 variance라고 볼 수 있다.** 고전적인 접근에서는 likeihood function의 값이 가장 큰 모델을 하나 고르는 방식(MLE)으로 문제를 풀고, 베이지안 접근에서는 likelihood function의 평균을 이용해서 inference를 한다.
<br/>
베이지안 접근에서는 likelihood function을 정의했으면, 그것을 이용하여 parameter의 posterior distribution을 구하고, 그것을 이용하여 likelihood function의 평균을 구함으로써 inference한다.
<br/>
**posterior**:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/2.png"></p>
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/4.png"></p>
식 2.2의 계산과정은 **marginalising likelihood over w**이고, **p(Y|X)는 model evidence라고 부른다.** Marginalisation은 간단한 모델(likelihood function과 prior distribution이 서로 conjugate)에서는 수식으로 계산이 가능하다. 그러나 그 외에는 수식으로 계산하기 힘들다. 심지어 not fixed basis function regression에서도 수식으로 계산이 힘들다. 따라서 Approximation은 필수적으로 요구된다. 
<br/>
**predictive distribution**:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/3.png"></p>
posterior을 구했다면 그것을 이용해서 marginalising likelihood over posterior을 해줘서 최종적으로 inference해준다.
<br/>
## 1.1.1 Variational inference 
앞서 말한 것 처럼 실제 posterior p(w|X,Y)는 수식으로 계산하기 힘들다. 때문에 posterior을 직접 구하는 것이 아닌 **variational distribution q(w)**를 정의하고 근사시킨다. q(w)는 가우시안 분포 같은 파라미터에 의해 정의되는 **parametric distribution**이다. KL divergence를 최소화하는 방법으로 근사시킨다.
<br/>
**KL divergence**:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/5.png"></p>
**model evidence P(Y|X)는 KL-divergence 항과 ELBO(evidence lower bound)의 덧셈으로 이루어진다.** 앞서 본 것처럼 P(Y|X)는 likelihood function의 marginalize이기 때문에 상수이고, 따라서 KL-divergence를 최소화하는 것은 EBLO를 최대화시키는 것과 동일하다.
<br/>
**model evidence & ELBO**:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/6.png"></p>
ELBO 항을 보면 왼쪽항은 log likelihood의 평균을 의미한다. 즉 이 값이 높을 수록 q(w)는 데이터를 더 잘 반영한다고 볼 수 있다. 그리고 두번째 항은 prior KL로 너무 현재 존재하는 데이터에 의존하지 않도록 해주는 regularize 항으로 볼 수 있다. 
<br/>
**predictive distribution**:
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/7.png"></p>
ELBO를 최대화 시키는 파라미터를 구했으면, 즉 최적의 variational distribution q(w)을 구했으면 그것을 통해 최종적으로 원하는 predictive distribution을 구할 수 있다. **피적분 함수는 likelihood function과 variational posterior의 곱으로 이루어진다.** posterior을 구하는 과정도 그렇고 predictive distribution을 구하는 과정도 그렇고 특정한 점 w에 의존하는 것이 아닌 전체적인 w를 고려하여 적분한다는 것에 주목하자. 전체적인 이 과정을 **variational inference**라 부른다. variational inference는 계산이 힘든 marginalization을 ELBO를 optimize하는 과정으로 대체하였고, bayesian modelling을 더 tractable하게 접근할 수 있게 되었다(model evidence(marginalization)의 적분 계산이 ELBO의 적분 계산보다 훨신 간단해 보이지만 optimize는 적분 계산이 필수적이지 않고 그 값을 최대화 시켜주는 파라미터 값만 찾기만 하면 되기 때문에 계산이 더 간단할 수 있다). **하지만 여전히 많은 데이터에 대해서는 ELBO 항의 적분에 대한 calculation을 하기가 힘들었고(모든 데이터에 대해서 적용되어야 한다. 물론 log likelihood라 덧셈으로 표현되겠지만), 복잡한 모델(q(w))에 대해서도 수식으로 계산하기가 힘들었다.**
</br>
## 2.2 Bayesian nueral networks
앞서 본 variational inference를 사용하더라도 많은 데이터, 복잡한 variational distribution에서는 ELBO 최적화가 어렵다. 이를 해결하기 위한 방법 중 하나로 딥러닝이 있고, 이것을 **bayesian neural networks, BNN)**이라 부른다. BNN은 기존 고전적인 베이지안 접근과 마찬가지로 모델의 모든 파라미터에 prior distribution을 설정하고 posterior을 구하는 방식으로 진행한다. BNN은 모델을 구성하는게 쉽다는 장점이 있지만, **inference 하는데 어렵다는 문제가 있다.** 여기서 말하는 inference가 어렵다는 것은 posterior을 구하는 것과 구한 posterior distribution을 모델에 적용하여 데이터를 입력받고 그것에 대한 출력 값을 구한는 과정(딥러닝에서의 inference)를 의마한다?








<br/>
<br/>
### <Hinton and Van Camp[1993] 작성 중>