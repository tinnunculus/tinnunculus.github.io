---
layout: post
title: compile wasm
sitemap: false
---

**참고**  
[1] <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/instantiate>  
[2] <https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/instantiateStreaming>  
* * *  

## 왜... 컴파일...?
> * wasm은 이미 C, C++ 같은 네이티브 언어가 컴파일된 것이다. 그럼에도 불구하고 사용하기 위해서는 또 컴파일 해야한다.
> * 기본적으로 wasm은 기계어(machine specific assembly)가 아니다. 그래서 바이너리 코드이긴 하지만 machine이 직접적으로 실행할 수 없는 코드이다.
> * 또한 wasm 자체적으로는 웹 API를 사용할 수 없다. 웹 API를 사용하기 위해서는 자바스크립트 코드를 사용해야 하기 때문에 자바스크립트 코드에서 돌아가야만 한다.
> * 그래서 wasm 코드는 자바스크립트에서 사용할 수 있도록 컴파일 되어야만 한다.
> * wasm 코드를 자바스크립트에서 사용할 수 있도록 컴파일하는 방법은 총 세가지이다.
> * WebAssembly.instantiate(), WebAssembly.instantiateStreaming(), WebAssembly.compileStraming()

## WebAssembly.instantiate()
> * .wasm 코드를 컴파일하고 인스턴스화 할 수도 있고, 이미 컴파일된 모듈을 인스턴스화 할 수도 있다.
> * .wasm 코드를 컴파일, 인스턴스화하기 위해서는 wasm 코드를 typed array 또는 ArrayBuffer의 형태로 취한 후 입력으로 넣어주면 된다. 반환된 Promise는 property로 컴파일된 WebAssembly.Moduel 및 인스턴스화된 WebAssembly.Instance를 가지고 있다.
> * 컴파일된 모듈을 인스턴스화 하기 위해서는 WebAssembly.Module을 입력으로 취하여 Instance를 resolved하는 Promise를 반환한다.
> * 또한 출력의 객체에 포함하고 싶은 property는 importObject 객체를 입력으로 넣어주면 되는데, 동일한 이름의 property가 있어야만 가능하다.
> * 하지만 동일한 역할을 하는데 Streaming 기능이 있는 WebAssembly.instantiateStreaming()을 사용하는 것이 좋다.
> * wasm 모듈 내부의 것들을 자바스크립트에서 쓸 수 있도록 할려면 우선 wasm 코드에서 export 코드를 통해 내부의 것을 밖으로 전달해야하고, 자바스크립트에서는 instance.exports를 통해 전달받을 수 있다.
> ~~~js
> // instatiate 메소드는 wasm 모듈을 컴파일과 동시에 인스턴스화를 한다.
> var import Object = { // instantiate 메소드의 importObject로 들어갈 객체 인스턴스랑 동일한 프로퍼티를 존재해야만 한다. "imports"
>     imports: {
>         imported_func: function(arg){
>             console.log(arg);
>         }
>     }
> };
> fetch('simple.wasm').then(response => 
>     response.arrayBuffer() // instantiate 함수는 arrayBuffer만 입력으로 받는다.
> ).then(bytes =>
>     WebAssembly.instantiate(bytes, importObject)
> ).them(result =>
>     result.instance.exports.exported_func()
> );
> ~~~
> ~~~js
> // worker
> // instantiate 메소드는 컴파일된 모듈을 입력으로 받으면 인스턴스화를 한다.
> var import Object = {
>     imports: {
>         imported_func: function(arg){
>             console.log(arg);
>         }
>     }
> };
> onmessage = function(e){
>     console.log('module received from main thread');
>     var mod = e.data;
>     WebAssembly.instantiate(mod, importObject).then(result =>
>         result.instance.exports.exported_func());
> };
> ~~~
> ~~~js
> // main thread
> var worker = new Worker("wasm_worker.js");
> WebAssembly.compileStraming(fetch('simple.wasm')).then(mod => // compileStraming 메소드는 .wasm 을 컴파일만 해준다.
>     worker.postMessage(mod));
> ~~~

## WebAssembly.instantiateStreaming()
> * WebAssemblt.instantiate()와 사용하는 방식은 동일하지만 스트리밍을 지원해서 ArrayBuffer가 아니라 fetch의 response를 입력으로 받는다.
> ~~~js
> var import Object = {
>     imports: {
>         imported_func: function(arg){
>             console.log(arg);
>         }
>     }
> };
> WebAssembly.instantiateStreaming(fetch('simple.wasm'), importObject)
> .then(obj => obj.instance.exports.exported_func());
> ~~~