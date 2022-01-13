---
layout: post
title: [node] require
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## require 메소드
> * require는 global 객체의 메소드이다.
> * require는 여러가지 역할을 한다.
> * 먼저 인자로 받은 자바스크립트 파일을 한번 실행한다.
> * 출력물로 인자로 받은 자바스크립트 파일의 module.exports 객체를 출력한다.

## require.cache
> * 현재 실행시킨 자바스크립트 파일을 main 모듈로 여기고, main 모듈에서 불러온 다른 모듈들을 저장한다.
> * 만약 모듈을 중복 require을 한다면 새로 require 하지 않고 requrie.cache에서 해당 모듈을 불러온다. 
> * 즉 중복 실행되지도 않고, 새로운 module.exports 객체를 가져오지도 않는다.
> * 자바스크립트는 스크립트 언어이기에 위에서 아래로 차례차례 실행된다는 것을 인지해야 한다.
> ~~~js
> // A.js
> A = {};
> module.exports = A;
> ~~~
> ~~~js
> // B.js
> A = require('./A');
> function test(){ console.log(A.c); }
> module.exports = test;
> ~~~
> ~~~js
> // C.js
> A = require('./A'); // A 파일은 실행되지 않는다. module에 대한 정보는 D의 require.cache에서 가져온다.
> A.c = 100;
> ~~~
> ~~~js
> // D.js
> const test = require('./B');
> console.log(require.cache); // 아래 출력물
> require('./C');
> test(); // 100
> ///// 출력물
> [Object: null prototype] {
>   '/Users/jongyeonlee/workspace/js_ex/D.js': Module {
>     id: '.',
>     path: '/Users/jongyeonlee/workspace/js_ex',
>     exports: {},
>     filename: '/Users/jongyeonlee/workspace/js_ex/C.js',
>     loaded: false,
>     children: [ [Module] ], // B
>     paths: [
>       '/Users/jongyeonlee/workspace/js_ex/node_modules',
>       '/Users/jongyeonlee/workspace/node_modules',
>       '/Users/jongyeonlee/node_modules',
>       '/Users/node_modules',
>       '/node_modules'
>     ]
>   },
>   '/Users/jongyeonlee/workspace/js_ex/B.js': Module {
>     id: '/Users/jongyeonlee/workspace/js_ex/D.js',
>     path: '/Users/jongyeonlee/workspace/js_ex',
>     exports: [Function: test],
>     filename: '/Users/jongyeonlee/workspace/js_ex/D.js',
>     loaded: true,
>     children: [ [Module] ], // A
>     paths: [
>       '/Users/jongyeonlee/workspace/js_ex/node_modules',
>       '/Users/jongyeonlee/workspace/node_modules',
>       '/Users/jongyeonlee/node_modules',
>       '/Users/node_modules',
>       '/node_modules'
>     ]
>   },
>   '/Users/jongyeonlee/workspace/js_ex/A.js': Module {
>     id: '/Users/jongyeonlee/workspace/js_ex/A.js',
>     path: '/Users/jongyeonlee/workspace/js_ex',
>     exports: {},
>     filename: '/Users/jongyeonlee/workspace/js_ex/A.js',
>     loaded: true,
>     children: [],
>     paths: [
>       '/Users/jongyeonlee/workspace/js_ex/node_modules',
>       '/Users/jongyeonlee/workspace/node_modules',
>       '/Users/jongyeonlee/node_modules',
>       '/Users/node_modules',
>       '/node_modules'
>     ]
>   }
> }
> ~~~