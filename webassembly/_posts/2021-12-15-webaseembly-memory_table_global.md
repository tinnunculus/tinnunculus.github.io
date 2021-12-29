---
layout: post
title: memory, table, global
sitemap: false
---

**참고**  
[1] <https://developer.mozilla.org/ko/docs/WebAssembly/Using_the_JavaScript_API>  
[2] <https://study.com/academy/lesson/what-is-the-linear-memory-model-definition-constraints.html>
* * *  

## memory instance
> * 웹어셈블리 모듈도 네이티브 C, C++과 같이 메모리 접근을 linear memory으로 표현한다.
> * linear memory model이란 전체 메모리 주소가 하나의 연속적인 값으로 표현되고 해당 주소에 **직접적으로 접근할 수 있는 모델**을 뜻한다.
> * linear memory model에서 데이터는 순차적(sequential)하고 연속적(continuous)하게 저장된다. (ex, 배열을 저장한다면 배열의 인덱스들은 메모리에 선형적으로 저장된다.)
> * C, C ++ 프로그램과 달리 WebAssembly 모듈의 인스턴스가 액세스 할 수 있는 메모리는 WebAssembly **메모리 객체**에 포함된 특정 범위로 제한됩니다. 즉 webassembly 모듈에서 **메모리에 접근하기 위해서는 메모리 객체를 만들어야 한다**.
> * 자바스크립트에서 Memory 인스턴스는 크기를 조정할 수 있는 ArrayBuffer로 생각할 수 있다.
> * 자바스크립트에서 **WebAssembly.Memory() Constructor**를 사용하여 웹어셈블리용? **Memory 객체**를 만들 수 있다.
> * WebAssembly.Memory()는 **ArrayBuffer를 반환하는 buffer accessible property**를 제공한다.
> * 메모리 인스턴스 생성시 최대 용량이 제공되었을 때 이 최대 값을 초과하여 접근하려고 시도하면 WebAssembly.RangeError 예외가 발생한다.
> ~~~js
> var memory = new WebAssembly.Memory({initial:10, maximum:100}); // 한 페이지에 64킬로 바이트
> console.log(memory.buffer); // memory.buffer getter method, ArrayBuffer(655360);
> new Uint32Array(memory.buffer)[0] = 42;
> new Uint32Array(memory.buffer)[0];
> new Uint32Array(memory.buffer);
> new Uint32Array(memory.buffer).slice(0, 10);
> ~~~
> * 메모리 인스턴스는 **Memory.prototype.grow()를 호출하여 확장**할 수 있다.
> * ArrayBuffer의 byteLength는 **불변**이므로, Memory.prototype.grow() 오퍼레이션이 성공하면, 버퍼 getter는 **새로운 ArrayBuffer 객체를 돌려준다**.
> ### 메모리 모듈 사용 예시
> > * 자바스크립트에서 웹어셈블리 모듈의 메모리에 접근하는 예시.
> > * 자바스크립트에서 WebAssembly.Memory constructor를 통해 메모리 객체를 생성한다.
> > * wasm 코드에서 메모리(js 객체의 mem property)를 import해서 자바스크립트에 있는 webassembly memory instance랑 wasm 코드의 메모리를 동기화하여 사용할 수 있다.
> > ~~~js
> > // memory.wasm
> > (memory (import "js" "mem") 1) // { js: { mem: memory } }
> > (func (export "accumulate") (param $ptr i32) (param $len i32) (result i32))
> > ~~~
> > ~~~js
> > const memory = new WebAssembly.Memory({initial:10, maximum:100}); // 자바스크립트에서 웹어셈블리에서 사용할 메모리 할당
> > WebAssembly.instantiateStreaming(fetch('memory.wasm'), { js: { mem: memory } }) // wasm 코드에서 import "js" "mem" 을 했으니 그거에 맞춰서 같이 넣어줘야 한다.
> > .then(results => {
> >     var i32 = new Uint32Array(memory.buffer); // ArrayBuffer
> >     for (var i = 0; i < 10; i++) {
> >         i32[i] = i;
> >     }
> >     var sum = results.instance.exports.accumulate(0, 10); // wasm 코드에서 exports 한 함수
> >     console.log(sum);
> > });
> > ~~~

## table instance
> * 자바스크립트에서 wasm의 함수를 사용하는 방법은 위의 accumulate 예시처럼 함수마다 exports를 하면 되지만, 더 편리하게 **여러 함수를 참조 형식으로 테이블로 묶어서 exports할 수도 있다**.
> * 참조는 안정성 및 이식성을 이유로 내용을 직접 읽거나 쓰지 않아야 하는 engine-trusted 값이므로 **함수에 대한 참조가 메모리에 저장되는 것은 안전하지 않다**.
> * **함수 참조는 함수 코드를 참조하므로 위에 언급한 안전상의 이유로 선형 메모리에 직접 저장할 수는 없다**. 대신 함수 참조는 테이블에 저장되며 선형 메모리에 저장할 수 있는 **정수인 인덱스**가 대신 전달된다. 
> * 함수 포인터를 호출 할 때, WebAssembly 호출자는 **인덱스를 제공한다**. 그 인덱스를 통해 함수에 접근할 수 있다.
> * linear memory model로 함수에 접근할 수 있으면 함수의 코드 하나하나에 접근할 수 있지만 인덱스를 통해 접근한다면 단순히 함수에만 접근할 수 있고 코드 한줄 한줄에는 접근하지 못하게 할 수 있다.
> * 그 호출자는 인덱싱된 함수 참조를 호출하기전에 테이블에 대해 safety bounds 검사를 할 수 있다.
> * Table은 Table의 값(함수) 중 하나를 불러오는 Table.prototype.get()
> * Table은 Table의 값(함수) 중 하나를 업데이트하는 Table.prototype.set()
> * Table에 저장할 수 있는 값(함수)의 수를 늘리는 Table.prototype.grow()
> ~~~js
> // table.wasm
> (func $thirteen (result i32) (i32.const 13))
> (func $fourtytwo (result i32) (i32.const 42))
> (table (export "tbl") anyfunc (elem $thirteen $fourtytwo))
> ~~~
> ~~~js
> WebAssembly.instantiateStreaming(fetch('table.wasm'))
> .then(function(results)){
>       var tbl = results.instance.exports.tbl;
>       console.log(tbl.get(0)()); // 0번 index로 thirteen 함수에 접근, table 안에 있는 함수의 코드를 실행시키고 싶으면 ()를 하면 된다.
>       console.log(tbl.get(1)()); // 1번 index로 fourtytwo 함수에 접근
> }
> ~~~

## globals
> * WebAssembly는 하나 이상의 컴파일된 인스턴스 전체를 자바스크립트에서 가져오기와 내보내기가 가능한 **하나의 전역 변수 인스턴스**를 생성할 수 있다.
> * 이는 여러 모듈을 동적으로 연결할 수 있으므로 매우 유용하다.
> * WebAssembly.Global constructor를 통해 생성하며, 이 생성자는 value, mutable(변형가능한지) property가 포함된 객체를 인자로 받는다.
> * value는 WebAssembly 현재 모듈의 인스턴스에서 사용하는 데이터 타입 i32, i64, f32, f64
> ~~~js
> // global.wasm
> (global $g (import "js" "global") (mut i32))
> (func (export "getGlobal") (result i32)
>     (global.get $g))
> (func (export "incGlobal")
>     (global.set $g
>         (i32.add (global.get $g) (i32.const 1))))
> ~~~
> ~~~js
> const global = new WebAssembly.Global({value:'i32', mutable:true}, 0);
> WebAssembly.instantiateStreaming(fetch('global.wasm'), { js: { global }})
> .then(result => {
>     let v = result.instace.exports.getGlobal(); // value = 0
>     global.value = 42;
>     v = result.instance.exports.getGlobal(); // value = 42
>     result.instace.exports.incGlobal(); // wasm 에서 export하는 함수로 global의 value 값을 1 증가시켜준다.
>     v = result.instance.exports.getGlobal(); // value = 43
> })
> ~~~