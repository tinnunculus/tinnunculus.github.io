---
layout: post
title: what is webassembly
sitemap: false
---

**참고**  
[1] <https://blog.logrocket.com/webassembly-how-and-why-559b7f96cd71>  
[2] <https://hacks.mozilla.org/2017/02/a-crash-course-in-assembly>  
[3] <https://hacks.mozilla.org/2017/02/creating-and-working-with-webassembly-modules>  
[4] <https://hacks.mozilla.org/2017/02/what-makes-webassembly-fast/>  
[5] <https://hacks.mozilla.org/2017/02/where-is-webassembly-now-and-whats-next/>
* * *  

* toc
{:toc}

## 어셈블리언어
> * 다양한 컴퓨터 machine이 존재하고, machine마다 다른 기계 명령어 셋(instruction set)을 가지고 있다.
> * machine이 이해할 수 있는 명령어는 사람이 이해할 수 없기 때문에 사람이 이해할 수 있는 언어로 매칭시킨 것을 어셈블리어(assembly)라고 한다.
> * 따라서 C나 C++ 언어 같은 high-level 언어를 컴파일하는 과정은 특정 언어와 특정 머신의 컴파일러가 따로 있어야 한다.
> * 하지만 이는 모든 언어와 모든 머신에대한 독립적인 컴파일러가 따로 만들어져야 한다는 단점이 있다.
> * 이를 어느정도 해결한 것이 GCC, LLVM 프로젝트이다. 이들은 획일화된 중간언어(IR)을 만들어서 해결한다.
> <p align="center"><img width="650" src="/assets/img/webassembly/what_webassembly/3.png"></p>

## 웹어셈블리언어란..?
> * 웹어셈블리어는 C, C++, Rust, swift, go 등 다양한 언어를 웹에서 실행시킬 수 있도록 돕는다.
> * 웹어셈블리언어는 위에서 설명한 어셈블리언어랑 다르다.
> * 특정 머신에 종속적인 어셈블리어와는 다르게 웹어셈블리언어는 특정 머신에 종속적이지 않다. 그렇다고 모든 머신에서 실행할 수 있는 것은 아니다. 웹어셈블리모듈만으로는 그 어떠한 머신에서도 실행시킬 수 없다. 그렇기 때문에 웹어셈블리 명령어를 virtual instruction이라고 부르기도 한다.
> * 웹어셈블리어는 어셈블리어의 개념보다는 high-level 언어와 어셈블리어(기계)의 중간단계라고 볼 수 있다.
> * 웹어셈블리 모듈 (.wasm)은 기본적으로 바이너리 포맷(binary)이다. 사람이 읽을 수 있게 텍스쳐 포멧(.wat)도 제공한다.
> * 웹어셈블리 모듈을 웹 어플리케이션에 넣을 수도 있고, 자바스크립트 파일에서 불러올 수도 있다.
> <p align="center"><img width="650" src="/assets/img/webassembly/what_webassembly/4.png"></p>

## 웹어셈블리어가 필요한 이유
> * 웹 브라우저는 그동한 HTML, CSS, JS만을 인정해왔다.
> * 자바스크립트는 초기에 간단한 하이퍼텍스트 문서를 제어하기 위해 등장했으며, 이것은 빨리 배우고 쉽게 사용할 수 있는 것에 초점을 맞추고 속도는 뒷전이었다.
> * 하지만 [JIT](https://tinnunculus.github.io/webassembly/2021-12-10-webassembly-jit/)같은 최적화 기법들도 나오면서 퍼포먼스 측면에서 개선이 되었지만, 여전히 비디오 스트리밍, 그래픽 처리 등에서 여러가지 한계가 있다. 이러한 분야에서 좀 더 최적화와 속도에 중심을 둔 네이티브 언어를 필요로 한다.
> * 웹어셈블리어는 C, C++ 같은 네이티브 언어를 웹에서 사용할 수 있도록 제공하며, 기존 자바스크립트 언어에 비해 빠르고 안정성있는 환경을 제공한다.
> * 웹어셈블리어를 직접적으로 다루지 못하더라도 npm을 통해서 다운로드하여 사용할 수도 있다.
> * 웹어셈블리 코드를 다운로드하고, 컴파일하고, 돌리는 일련의 과정을 온전히 자바스크립트로 제어할 수 있다.
> * 미래에는 웹어셈블리 모듈이 (script type='module'을 사용해서) ES2015모듈처럼 로드 가능하게 될 거다.


## 웹어셈블리 모듈은 어떻게 만들어지나
> * C, C++ 소스 코드를 LLVM 베이스의 high-level 컴파일러(Clang)을 사용하여 LLVM IR로 컴파일한다. (Rust언어는 자체 컴파일러인 rustc를 사용해서 wasm 파일을 만들 수 있다)
> * LLVM IR 코드에서 LLVM 최적화를 한다.
> * LLVM IR 코드를 wasm 컴파일러를 사용하여 wasm 모듈을 생성한다.
> * 하지만 wasm 컴파일러를 사용하는 방법은 최적화와 개발툴이 많이 않다.
> * LLVM IR 코드를 Emscripten 컴파일러를 사용하여 asm.js 파일로 컴파일하고 asm.js파일을 wasm 모듈로 컴파일하는 방법을 쓸 수 있다.

## 자바스크립트 vs 웹어셈블리 파이프라인
> * 자바스크립트의 파이프라인
> <p align="center"><img src="/assets/img/webassembly/what_webassembly/9.png"></p>
> <br/>
> * 웹어셈블리의 파이프라인
> <p align="center"><img src="/assets/img/webassembly/what_webassembly/10.png"></p>
> ### Fetching
> > * 소스 코드를 로딩하는 과정.
> > * 웹어셈블리는 자바스크립트보다 훨신 컴팩트해서 로딩속도가 훨신 빠르다.
> > * 비록 압축 알고리즘이 자바스크립트 코드의 크기를 많이 줄이긴 하지만 웹어셈블리는 압축된 바이너리 포맷이라 더 작을 수 밖에 없다.
>
> ### Parsing
> > * 텍스트인 소스 코드를 AST(Abstract Syntax Tree)로 바꾸는 과정
> > * 웹어셈블리 코드는 바이너리 포멧이기 때문에 Parsing 과정이 필요없다.
>
> ### Compiling + Optimizing
> > * 자바스크립트는 인터프리터 기반으로 컴파일되기 때문에 [JIT](https://tinnunculus.github.io/webassembly/2021-12-10-webassembly-jit/)에서 본 것처럼 컴파일과 최적화과정이 쉽지 않다.
> > * 웹어셈블리 모듈은 이미 머신 코드에 근접해있기 때문에 이러한 문제를 해결할 수 있다.
> > * 웹어셈블리 모듈은 더이상 변수의 타입을 체크하는데 시간을 쓸 필요가 없다. JIT의 최적화 과정 중에 만들어지는 여러 버전의 코드를 만들 필요가 없다.
> > * 이미 LLVM이 최적화를 많이 진행한 상태이기 때문에 적은 컴파일과 최적화만을 필요로 한다.
>
> ### Reoptimizing
> > * JIT는 reoptimize 과정이 필요하다. JIT의 optimize과정에서 잘못된 부분이 있을 수 있기 때문에 reoptimize 과정을 필요로 하다.[JIT](https://tinnunculus.github.io/webassembly/2021-12-10-webassembly-jit/) 참고
> > * 웹어셈블리에서는 타입이 정해져있기 때문에 JIT가 더이상 타입을 가정(optimization)할 필요가 없고 이것은 reoptimization 과정이 필요 없다.
>
> ### Garbage collection
> > * 자바스크립트에서는 개발자들은 우리가 생성한 변수들의 메모리에 대해서 신경 쓸 필요가 없다. JS 엔진이 자동적으로 관리를 해주기 때문이다.
> > * 실행과정에서 보면 이러한 과정은 속도를 빠르게 해주기 위한 것 보다는 사용자의 편의성을 위한 과정이며, 속도를 저해시키는 요인 중 하나이다.
> > * 웹어셈블리 모듈에서는 이러한 것을 제공하지 않는다. 모든 메모리 관리를 사용자에게 맡긴다.
> > * 이것은 프로그래밍을 어렵게 만드는 요인중 하나지만 퍼포먼스 측면에서는 이점이다.

## 웹어셈블리 모듈 어떻게 사용하나
> * 웹어셈블리 모듈 하나만으로는 브라우저에서 실행시킬 수 없다.
> * 웹어셈블리언어는 DOM이나 WebGL, WebAudio 같은 플랫폼 API에 직접적인 접근을 할 수 없다. 단지 자바스크립트를 호출하면서 정수나 부동소수점 기초 자료형을 넘겨줄 수 있을 뿐이다.
> * 웹 브라우저에 웹어셈블리 코드를 적용시키기 위해서는 자바스크립트를 통해서만 가야한다.
> * Emscripten 컴파일러는 컴파일 결과물로 wasm 뿐만 아니라 자바스크립트, HTML 파일도 같이 출력한다.
> * Emscripten이 생성하는 자바스크립트 코드는 웹어셈블리 모듈을 로드하고 이것을 웹 API와 소통하는것을 가능하게 한다.
> * Emscripten이 생성하는 HTML 코드는 해당 자바스크립트 파일을 로드하고 웹어셈블리 모듈을 display하는 것을 가능하게 한다.
> * Emscripten은 자바스크립트와는 다르게 virtual 파일 시스템을 이용해서 local 파일 시스템에 접근을 허용한다.
> * 웹어셈블리 모듈은 자바스크립트 엔진에서 자바스크립트와 같이 머신에 specific한 어셈블리어로 컴파일된다.
> <p align="center"><img src="/assets/img/webassembly/what_webassembly/6.png"></p>

## Emscripten의 기능
> * Emscripten에 대해서는 따로 자세히 다룰 것이므로 여기서는 간단한 역할 소개만 한다.
> * Emscripten은 POSIX의 일부분과 SDL, OpenGL, OpenAL같은 유명 C/C++ 라이브러리를 직접 구현했다.
> * 이 라이브러리들은 웹 API 위에서 구현되어야 하는데 웹 API에 웹 어셈블리를 연결시켜주는 자바스크립트 접착제(glue) 코드가 각각의 라이브러리에 있어야한다.
> * Emscripten을 통해 생성된 HTML 문서는 자바스크립틑 접착제 코드를 불러오고 표준출력(stdout)을 <textarea>에 작성한다. 만약 애플리케이션이 OpenGL을 사용하고있으면 HTML 안에 렌더링 타겟으로 사용되는 <canvas> 엘리먼트가 포함됩니다.

## 웹어셈블링의 특징
> * 웹어셈블리어는 함수의 parameter나 return 데이터로 int16 int32 float16 float32만을 사용할 수 있다.
> * 만약 문자열 데이터를 사용하고 싶다면 숫자 배열로 바꾼 뒤 메모리 접근을 통해야만 한다.
> * 이를 행하기 위해서는 웹어셈블리 모듈의 ArrayBuffer를 이용해야 한다. [자세한 내용](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_objects/WebAssembly/Memory)
> * 자바스크립트와 웸어셈블리는 메모리를 카피하거나 call stack을 불러와야만 접근이 가능하다. 변수를 통한 접근은 불가능하다.
> * 웹어셈블리모듈은 기본적으로 스택 머신(stack machine)이다. 
> * 스택 머신은 값을 만날때 마다 스택에 쌓은 후, 연산자를 만날 때 마다 스택에서 값을 꺼내 쓰는 방식이다. 다른 어셈블리어와 다르게 레지스터에 값을 등록하지 않는다.
> * 이것이 가능한 이유는 웹어셈블리는 머신에 직결되지 않는다. 그렇기 때문에 레지스터를 등록할 필요가 없다.
> * 이러한 방식은 웹어셈블리 모듈을 더욱 가볍게 만들고 로딩하는데 걸리는 시간을 줄여준다.
> * 실제 자바스크립트 엔진이 어셈블리어로 컴파일할 때는 레지스터로 등록하여 사용한다.
> * <http://mbebenita.github.io/WasmExplorer/> 여기서 C 소스코드를 wat(wasm)코드와 어셈블리어(기계어)로 번역하는 것을 볼 수 있다.
> ~~~c++
> int add42(int num){
>     return num + 42;
> }
> ~~~
> <p align="center"><img src="/assets/img/webassembly/what_webassembly/8.png"></p>
> * wasm 모듈에는 section이라고 불리는게 있다.
> * Required section : Type, Function, Code
> * Option section : Export, import, Start, Global, Memory, Table, Data, Element
> * 자바스크립트 API는 모듈, 메모리, 테이블, 인스턴스를 생성하는 방법을 제공한다.
> * Section을 사용하는 자세한 방법은 따로 알아보도록 한다.