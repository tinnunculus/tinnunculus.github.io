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
> * 기본적으로 wasm은 기계어(machine specific assembly)가 아니다. 그래서 바이너리 코드이긴 하지만 machine이 직접적으로 실행할 수 없는 코드이다.
> * 또한 wasm 자체적으로는 웹 API를 사용할 수 없다. 웹 API를 사용하기 위해서는 자바스크립트 코드를 사용해야 하기 때문에 자바스크립트 코드에서 돌아가야만 한다.
> * 그래서 wasm 코드는 순수한 자바스크립트 코드가 아니기 때문에 바로 자바스크립트 엔진이 wasm 코드를 이해할 수는 없다. 즉 자바스크립트에서 사용할 수 있도록 컴파일 되어야만 한다.
> * 결론적으로는 자바스크립트 엔진이 wasm 코드를 만났을 때는 이미 컴파일된 기계어가 되어 있어서 바로 실행시킬 수 있도록 한게 아닐까?
> * wasm 코드를 자바스크립트에서 사용할 수 있도록 컴파일하는 방법은 총 세가지이다.
> * WebAssembly.instantiate(), WebAssembly.instantiateStreaming(), WebAssembly.compileStraming()

## WebAssembly.instantiate()
> * .wasm 코드를 컴파일하고 **인스턴스화** 할 수도 있고, 이미 컴파일된 모듈을 인스턴스화 할 수도 있다. (인스턴스는 직접적으로 사용할 수 있도록 객체화 한다는 뜻인듯)
> * .wasm 코드를 컴파일, 인스턴스화하기 위해서는 wasm **코드**를 typed array 또는 ArrayBuffer의 형태로 취한 후 입력으로 넣어주면 된다. 반환된 Promise는 property로 컴파일된 WebAssembly.Module 및 인스턴스화된 WebAssembly.Instance를 가지고 있다.
> * 컴파일된 모듈을 인스턴스화 하기 위해서는 WebAssembly.Module을 입력으로 취하여 Instance를 resolved하는 Promise를 반환한다.
> * **wasm 코드에서 import**하는 변수나 함수나 객체는 **importObject**를 통해서 전달한다.
> * wasm 코드에서 export하는 변수나 함수나 객체는 instance.exports 를 통해서 접근한다.
> * 하지만 동일한 역할을 하는데 Streaming 기능이 있는 WebAssembly.instantiateStreaming()을 사용하는 것이 좋다.
> ~~~js
> var import Object = {
>     imports: {
>         imported_func: function(arg){
>             console.log(arg);
>         }
>     }
> };
> fetch('simple.wasm').then(response => 
>     response.arrayBuffer() // 코드를 arrayBuffer로 변환시킨다. wasm과 자바스크립트가 데이터를 주고 받기 위해서는 메모리를 통해 해야만 한다.
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
> WebAssembly.compileStraming(fetch('simple.wasm')).then(mod =>
>     worker.postMessage(mod));
> ~~~

## WebAssembly.instantiateStreaming()
> * WebAssemblt.instantiate()와 사용하는 방식은 동일하지만 스트리밍을 지원해서 **ArrayBuffer로 한번에 받는게 아닌 fetch의 response를 입력으로 받는다**.
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