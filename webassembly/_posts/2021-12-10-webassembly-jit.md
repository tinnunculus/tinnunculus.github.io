---
layout: post
title: just-in-time compilers
sitemap: false
---

**참고**  
[1]. <https://hacks.mozilla.org/2017/02/a-crash-course-in-just-in-time-jit-compilers>  
* * *  

## 초기의 자바스크립트 엔진
> * 브라우저와의 실시간 피드백과 빠른 개발을 위해, 자바스크립트가 처음 나왔을 당시에는 인터프리터를 이용하여 컴파일 하엿다.
> * 인터프리터를 사용할 경우 코드를 보고, 수정하는데 있어서 컴파일러보다 용이하기 때문에 개발 속도가 더 빠르다.
> * 순수한 인터프리터를 사용할 경우 단순 반복되는 코드를 항상 컴파일하기 때문에 반복문과 같은 코드를 처리할 때, 속도적인 측면에서 매우 느리게 처리되는 단점이 있엇다.
> * 인터프리터는 런타임에 실시간으로 컴파일을 진행하기 때문에 코드 최적화를 하는데 어렵다.
> * 하지만 자바스크립트의 개발 초기에는 빠른 처리 속도보다는 빠른 개발 속도를 중요시 여겼다.

## just in time compilers의 등장
> * 초기 자바스크립트의 엔진으로 사용되었던 인터프리터의 비효율성을 해결하기 위해 등장하였다.
> * 기존의 인터프리터의 방식은 유지하되, monitor(aka profiler)라고 부르는 것을 자바스크립트 엔진에 포함시켰다.
> * monitor는 현재 기본 인터프리터를 통해 실행중인 코드를 보면서, 어떤 코드들이 몇번 반복 사용되는지, 데이터들의 타입은 어떠한 지 확인한다.
> * 몇번 반복되는 코드 라인을 발견하면, 해당 코드를 **warm**이라고 지칭하고, 더 많이 반복되면 **hot**이라고 지칭하였다.
> <center><img src="/assets/img/webassembly/jit/1.png" height="100" width="50"></center>

## Monitor
> * 자바스크립트 코드의 런타임 중에 어떤 함수가 실행되면서 warm 코드를 만난다면, monitor는 해당 코드를 컴파일하라고 보내고, 컴파일된 코드를 **stub**이라고 부른다.
> * stub은 해당 코드의 줄(line)과 변수 타입(variable type)으로 인텍스화 되어있다.
> * 만약 런타임 중에 모니터가 해당 코드를 같은 변수 타입으로 만난다면 단순히 컴파일된 코드를 꺼내온다.
> * 이것은 자바스크립트의 실행 속도를 증가시키지만, hot한 코드들이 많다면 더 강한 최적화가 필요하다.

## Optimizing Compiler
> * 어떤 코드가 warm을 넘어서 hot하다면, monitor는 해당 코드를 **optimizing compiler** 한테 보낸다.
> * optimizing compiler는 해당 코드(함수)를 빠르게 실행될 수 있도록 최적화를 진행하고 컴파일하여 저장한다.
> * 최적화 과정은 어쩔 수 없이 여러 가정을 하게된다. 물론 그 가정들은 monitor가 관찰한 근거에 의한다. 예를들면 특정한 constructor에서 생성한 객체들은 모두 같은 모양(same properties, mothods 등등)을 띄고 있다고 가정하고 최적화를 진행한다.
> * 그 가정이 항상 맞다고 보장하지는 않는다. 99개의 객체가 같은 모양을 띄지만 하나의 객체가 다른 모양을 띌 수도 있다.
> * 컴파일러는 해당 코드를 실행하기 전에 타당성(valid) 체크를 하고 타당하지 않다면 해당 코드를 버리고 다시 베이스 라인 컴파일러 버젼으로 돌아간다. 이러한 과정을 **deoptimization** 이라고 한다.
> * optimizing compiler는 대체적으로 코드를 빠르게 한다. 하지만 deoptimization 같은 문제 때문에 종종 퍼포먼스를 악화시키기도 한다.
> * 이처럼 실시간 컴파일에서 코드 최적화를 하는 것은 쉽지 않다.

## 최적화 예시: Type specialization
> * Optimizing Compiler가 최적화하는 방법들은 다양한 방법이 있지만 가장 큰 영향력을 끼치는 방법중 하나가 **Type specialization**이다.
> * 자바스크립트에서 시행하는 dynamic type system은 런타임에 더 많은 일을 요구한다.
> ~~~js
> function arraySum(arr){
>     var sum = 0;
>     for (var i = 0; i < arr.length; i++){
>         sum += arr[i];
>     }
> }
> ~~~
> * 위의 코드는 실행되면 monitor에 의해서 warm up 될 것이고, baseline compiler는 해당 코드들의 stub를 생성할 것이다.
> * sum += arr[i]의 stub에서 += operation은 정수형 더하기로 다뤄질 것이다.
> * 하지만 sum이랑 arr[i]의 더하기는 항상 정수형이라는 것을 보장하지는 못한다. 마지막 인수의 덧셈만 실수형일 수도 있을테니 말이다.
> * JIT는 위와 같은 문제를 다루기 위해 하나의 코드에 대해서 여러개의 stub들을 만들고 어떤 stub를 고를지 선택할 수 있게 여러 질문 또한 만든다.
> <center><img src="/assets/img/webassembly/jit/2.png" height="100" width="50"></center>
> * 각각의 코드들은 stub의 집합들을 가지고 있다.
> * JIT는 여전히 코드의 타입을 알기위해 코드를 만날때마다 여러 질문들을 체크해야만한다.
> <center><img src="/assets/img/webassembly/jit/3.png" height="100" width="50"></center>
> * optimzing compiler는 매번 반복해서 타입 체크해야하는 과정을 없애 코드 최적화를 한다.
> * 라인별로 컴파일하는 것이 아닌 해당 함수(warm up된)를 한번에 컴파일 하는데 타입체크도 미리 한번에 한다.
> <center><img src="/assets/img/webassembly/jit/4.png" height="100" width="50"></center>