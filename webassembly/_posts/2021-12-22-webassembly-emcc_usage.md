---
layout: post
title: emcc simple usage
sitemap: false
---

**참고**  
[1] <https://emscripten.org/docs/getting_started/Tutorial.html>  
* * *  

## 간단한 emcc 컴파일러 사용법
> * emsdk_env.sh 스크립트를 source 해야 emcc 컴파일러를 사용할 수 있다.
> * emcc로 C, C++ 코드를 컴파일하면 자바스크립트 파일, wasm 파일, html 파일이 출력된다.
> * 자바스크립트 파일은 wasm 코드를 자바스크립트에서 로딩하고 실행할 수 있도록 해주는 코드이다.
> * 자바스크립트 파일은 node로 실행시킬 수 있다.
> * 웹 브라우저에서 뛰워볼 수 있도록 html 파일도 제공한다. -o 옵션을 사용하면 된다.
> * html 파일을 브라우저에서 열기 위해서는 http 요청으로만 열어야 한다.
> * python3 -m http.server 을 통해 http 서버를 실행하고 html 파일을 요청해서 로컬 기계에서 열 수 있다.
> ~~~
> emcc hello_world.c -o hello.html
> ~~~

## Using files
> * 자바스크립트는 로컬 파일 시스템에 직접적인 접근을 할 수 없다.
> * Emscripten은 로컬 파일 시스템을 virtual file system으로 simulate 할 수 있어서 우리는 C, C++의 fopen, fclose 같은 stdio를 사용하여 로컬 파일에 접근할 수 있다.
> * 우리가 접근하고 싶은 로컬 파일을 virtual file system 에 preloaded 해야한다.
> * 로컬 파일의 preloaded는 compile time에 진행되어야 한다.
> * compile time에 파일들을 미리 로딩하는 것은 자바스크립트의 파일 시스템이 asynchronous인 synchronous하게 미리 로딩한다는 장점이 있다.
> * 미리 로컬 파일을 로딩을 해놓기 때문에 자바스크립트에서의 코드들은 바로 접근할 수 있다.
> ~~~
> emcc hello_world.c -o hello.html --preload-file hello_world.txt
> ~~~

## Optimizing code
> * gcc나 clang 같이(실제로 clang 사용) optimization version을 제공한다.
> * -01, -02, -03 이 있으며 숫자가 높아질수록 강한 optimization을 해준다.
> * 예를 들면 -01 버전에서 printf는 puts 코드로 바뀐다.
> ~~~
> emcc -01 hello_world.c -o hello.html --preload-file hello_world.txt
> ~~~