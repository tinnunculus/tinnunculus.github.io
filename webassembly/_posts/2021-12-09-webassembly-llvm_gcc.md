---
layout: post
title: LLVM vs GCC
sitemap: false
---

**참고**  
[1] <https://www.omnisci.com/technical-glossary/llvm>  
[2] <https://stackoverflow.com/questions/24836183/what-is-the-difference-between-clang-and-llvm-and-gcc-g>  
[3] <https://d2.naver.com/helloworld/8257914>  
* * *  

* toc
{:toc}

## GCC
> * GCC(GNU Compilert Collection)의 약자로 GNU 프로젝트의 일환으로 개발된 **컴파일러 및 관련 툴들의 모음**이다.
> * GCC는 frontend에서는 C++, java, fortran, ada, go 등 수 많은 여러 언어를 컴파일 할 수 있다. backend에서는 intel x86, ARM 등 다양한 머신의 기계어로 컴파일 할 수 있다.
> * GCC는 언어, 머신마다 **다른 컴파일러**를 가지고 있고, 각각에 **종속적으로** 최적화를 시킨다.
> * 자체 중간 언어인 **GIMPLE, RTL**(Register Transfer Language)을 가지고 있다.
> * C 소스 코드가 있다면, frontend에서 GCC의 C 컴파일러가 RTL로 바꿔주고, backend에서 GCC의 특정 머신(intel x86, ARM)에 맞는 컴파일러가 해당 기계어로 바꿔준다.
> <p align="center"><img width="550" src="/assets/img/webassembly/llvm_gcc/1.png"></p>

## LLVM
> * GCC과 마찬가지로 LLVM 또한 컴파일에 관련된 기술들을 개발하는 프로젝트이다.
> * 다만 단순 컴파일과 그 최적화만 추구하는 GCC와 달리 LLVM은 그 자체로 **재사용(reusable) 가능성**을 추구한다.
> * LLVM은 여러 언어, 플랫폼, 아키텍쳐에서 코드들이 폴팅(porting)될 수 있도록 노력한다.
> * 자체 중간 언어인 **LLVM IR(Intermediate Repersentation)**을 가지고 있다.
> * C 소스 코드가 있다면, frontend에서 Clang 컴파일러가 해당 소스 코드를 LLVM IR으로 바꿔주고, backend에서 특정 머신에 해당하는 LLVM 컴파일러가 해당 머신의 기계어로 바꿔준다.
> * LLVM은 GCC와는 다르게 특정 플랫폼, 특정 언어에 대한 의존성이 낮다.
> * 그 예시로 Clang은 최적화보다 재사용성과 안정성을 추구하고, 최적화는 LLVM에 의존하도록 설계되었다. 이렇게 하면 LLVM IR 코드는 사용자에 의해 재사용 및 재가공이 될 수 있으며 이는 특정 언어에 대한 독립성을 추구할 수 있다는 뜻이다. (GCC는 프론트엔드에서부터 각각의 개별적인 최적화를 통해 중간언어에서 보면 많은 변형과 읽기 불가능한 코드를 생성할 수 있다)
> * GCC가 LLVM보다 빠르다고 전해졌지만, 최근에는 많이 따라왔다.
> * GCC가 전통적인 많은 언어를 제공하지만 최신에 나온 언어들은 LLVM이 제공하는 경우도 많다.
> <p align="center"><img width="550" src="/assets/img/webassembly/llvm_gcc/2.png"></p>