---
layout: post
title: wat
sitemap: false
---

**참고**  
[1] <https://developer.mozilla.org/ko/docs/WebAssembly/Understanding_the_text_format>  
[2] <https://developer.mozilla.org/ko/docs/WebAssembly/Text_format_to_wasm>
* * *  

## S-expressions
> * wat의 기본적인 텍스트 구조는 **S-expressions**이다.
> * S-expressions는 **트리**를 텍스트 형식으로 방법이다.
> * Abstract Tree와는 다르게 단순하면서 일반적으로 많이 사용하는 방식으로 구성되어 있다.
> * 트리의 각 노드는 괄호를 통해 나타낸다.
> * 아래 wat 코드는 모듈이라는 최상위 노드와 2개의 자식 노드를 가진 트리이다.
> ~~~js
> (module (memory 1) (func))
> ~~~
> * wat의 **모든 코드**는 다음과 같은 구조를 갖는 **함수로** 구성되어있다.
> ~~~js
> (func <signature> <locals> <body>)
> ~~~

## Signatures and parameters
> * 함수의 parameter와 return의 타입을 지정한다. (param result 순으로 배치한다)
> * 현재는 단 하나의 반환 타입을 가질 수 있다.
> * result 노드가 없다면 아무것도 반환하지 않는다.
> * i32, i64, f32, f64 타입만 가능하다.
> * 아래의 코드는 2개의 32bit int를 받고 64bit float를 반환하는 함수이다.
> ~~~js
> (func (param i32) (param i32) (result f64) ...)
> ~~~
> * 함수 내부에서 사용할 **local 변수**를 설정할 수도 있다.
> * signatures 뒤에 (local type)을 통해서 선언한다.
> ~~~js
> (func (param i32) (param i32) (result f64) (local i32) ...)
> ~~~

## local와 parameter를 getting, setting 하기
> * get_local 명령어와 set_local 명령어로 함수의 parameter와 local 변수를 읽고 쓸 수 있다.
> ~~~js
> (func (param i32) (param f32) (local f64)
>   get_local 0  // param i32 매개변수를 받는다.
>   get_local 1  // param f32 매개변수를 받는다.
>   get_local 2) // local f64 매개변수를 받는다.
> ~~~
> * 위의 코드처럼 index를 통해서 변수를 받지 않고, **이름**을 통해서도 받을 수 있다.
> ~~~js
> (func (param $p1 i32) (param $p2 f32) (local $loc f64)
>   get_local $p1
>   get_local $p2
>   get_local $loc)
> ~~~

## stack machines
> * 앞서 본 코드에서 보면 함수의 지역변수를 받아올 때 따로 저장하는 레지스터를 등록하지 않았다. 이것은 webassembly가 기본적으로 **stack machine**이기 때문이다.
> * **모든 데이터를 스택에 넣고 빼는 방식**으로 진행된다. get_local 명령어를 통해 스택에 넣는다.
> * 스택은 함수마다 하나의 스택을 사용한다고 생각해야한다.
> * 리턴 데이터가 하나이면 스택에 하나의 데이터가 존재해야하고 리턴하지 않으면 빈스택으로 끝 맞춰야 한다.
> ~~~js
> (func (param $p1 i32) (param $p2 i32) (result i32) // result i32 이므로 마지막에 스택에는 i32형 데이터 하나가 남아있어야 한다.
>   get_local $p1
>   get_local $p2
>   i32.add)
> ~~~

## 함수 호출하기
> * 변수와 마찬기지로 함수 또한 인덱스를 통해서 식별되지만 이름을 붙일 수도 있다.
> ~~~js
> (func $add ...)
> ~~~
> * 함수를 외부에서 사용하기 위해서는 해당 함수를 export해야만 한다.
> ~~~js
> (module
>   (func $add (param $p1 i32) (param $p2 i32) (result i32)
>     get_local $p1
>     get_local $p2
>     i32.add)
>   (export "add" (func $add)) // add라는 이름으로 add 함수를 export한다.
> )
> ~~~
> * 위의 코드를 자바스크립트 코드에서 사용하기 위해서는 instance.exports.function을 하면된다.
> ~~~js
> WebAssembly.instantiateStreaming(fetch('add.wasm'))
> .then(resolve => {
>     result = resolve.instance.exports.add(1, 2); // result = 3
> });
> ~~~

## 같은 wat 코드 내에서 다른 함수 호출하기
> * wat의 call 명령어는 모듈 내에서 단일 함수를 호출한다.
> ~~~js
> (module 
>     (func $getAnswer) (result i32)
>         i32.const 42) // stack에 42를 쌓는건가??
>     (func (export "getAnswerPlus1") (result i32) // function의 정의와 함께 export 함수임을 알릴 수 있다. 요게 편한듯...
>         call $getAnswer // 함수 getAnswer 호출, stack에 42가 쌓임
>         i32.const 1 // stack에 1이 쌓임
>         i32.add)) // 42 와 1 
> ~~~

## 자바스크립트에서 함수 가져오기
> * wat 모듈에서 자바스크립트 함수를 불러올 수도 있다... 어떻게 가능한건지는 모르겟지만... 그 함수인 부분은 자바스크립트 코드를 실행시킬듯...
> * 아래 코드에서 "console" "log" 은 두단계의 네임스페이스를 의미하여 console 모듈의 log 함수를 가져 오기를 요청한다.
> * 가져온 함수를 (func ...)을 통해 선언해줘야 한다. 
> ~~~js
> (module
>     (import "console" "log" (func $log (param i32)))
>     (func (export "logIt")
>         i32.const 13 // stack에 13 push
>         call $log)) // stack에 있는 13 입력으로
> ~~~
> * 자바스크립트에서는 wat 모듈로 함수를 보낼려면 instantiate 함수에 importObject를 인수로 넣어야 한다.
> ~~~js
> const importObject = {
>     console: {
>         log: function(arg){ // int32 만 가능
>             console.log(arg);
>         }
>     }
> };
> WebAssembly.instantiateStreaming(fetch('logger.wasm'), importObject)
> .then(obj => {
>     obj.instance.exports.logIt();
> });
> ~~~

## wat 모듈에서의 전역 변수
> * wasm은 하나 이상의 모듈에서 사용가능하고 자바스크립트에서 읽기와 쓰기가 가능한 전역 변수를 선언할 수 있다.
> * 자바스크립트에서는 WebAssembly.Global() 생성자를 사용하여 접근해야한다.
> * 자바스크립트에서 접근하기 위해서는 반드시 import를 해야한다.
> ~~~js
> (module
>     (global $g (import "js" "global") (mut i32)) // 자바스크립트에서 wasm의 전역변수를 사용하고 싶다면 import를 해줘야한다.
>     (func (export "getGlobal") (result i32)
>         (get_global $g)) // g global 변수를 얻어서 스택에 저장.
>     (func (export "incGlobal")
>         (set_global $g
>             (i32.add (get_global $g) (i32.const 1)))) // add 함수가 그냥 쌩으로 있으면 스택에서 가져와서 쓰지만 이렇게도 쓸 수 있다. 즉석으로 스택에 넣고 빼기??
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

## wat 모듈에서의 momory
> * wasm에는 linear memory 에서 여러 데이터를 한번에 읽고 쓰는 데 필요한 i32.load 및 i32.store 를 제공한다.
> * 자바스크립트 **WebAssembly.Memory** 생성자를 통해서 자바스크립트와 wasm 코드간에 메모리 접근을 **공유**한다.
> * 자바스크립트에서 생성한 메모리를 wasm에서 사용하기 위해서는 import를 해야한다.
> * 할당된 메모리는 **data section**에 지정되며, 그래서 data section에 데이터를 저장하면 된다.
> ~~~js
> (module
>     (import "console" "log" (func $log (param i32 i32))) // 이런식으로 (param i32) (param i32) 를 (param i32 i32)으로 써도 되는듯.
>     (import "js" "mem" (memory 1)) // 메모리를 import 하는데 메모리의 최소 크기는 1페이지이다. 라는뜻..
>     (data (i32.const 0) "Hi") // data section의 0번째 주소부터 "Hi" 등록
>     (func (export "writeHi")
>         i32.const 0
>         i32.const 2
>         call $log))
> )
> ~~~
> ~~~js
> function consoleLogString(offset, length){
>     let bytes = new Uint8Array(memory.buffer, offset, length); // 자바스크립트는 인터프리터 언어니까 memory 객체 사용가능.
>     let string = new TextDecoder('utf8').decode(bytes); // byte를 utf-8로 ??
>     console.log(string);
> }
> let memory = new WebAssembly.Memory({initial:1});
> let importObject = {console : { log :consoleLogString }, js: {mem: memory}};
> WebAssembly.instantiateStreaming(fetch('test.wasm'), importObject)
> .then(result => {
>     result.instance.exports.writeHi();
> })
> ~~~

## wat 모듈에서의 table
> * wasm에서는 anyfunc 옵션을 추가할 수 있다. function의 signature가 어떠한 타입도 받을 수 있다는 뜻.
> * 하지만 anyfunc은 보안상의 이유로 linear memory에 저장할 수 없다.
> * linear memory는 original address를 사용하므로 임의로 접근을 혀용하면 안될 일이다.
> * 해결책으로 나온 것이 테이블에 함수 참조를 저장하고 테이블 인덱스를 전달하는 것으로 메모리 주소를 유출하는 것을 피할 수 있다.
> * linear memory를 data 섹션을 이용해서 초기화 하는 것처럼 elem 섹션을 사용하여 함수가 있는 테이블 영역을 초기화 할 수 있다.
> ~~~js
> (module
>     (table 2 anyfunc) // 2는 테이블의 초기 크기, anyfunc는 any signature의 함수.
>     (elem (i32.const 0) $f1 $f2) // i32.const 0 은 인덱스의 시작부분을 나타낸다. f1, f2 함수를 테이블에 등록한다. 각각 인덱스는 0과 1
>     (func $f1 (result i32)
>         i32.const 42)
>     (func $f2 (result i32)
>         i32.const13)
> )
> ~~~

## table 사용 시에 타입 체크하는 법
> * 테이블에 지정된 함수의 input output type들을 체크할 수 있는 방법은 **type 노드**를 사용하는 것이다.
> * 아래 코드는 함수가 i32 type을 리턴으로 하는지 체크하는 것이다.
> * call_indirect 함수는 stack에서 값을 하나 pop하여 해당 값을 index로 **table에 있는 함수에 접근하여 call** 한다.
> * 하나의 모듈에는 하나의 테이블만이 존재할 수 있다.
> ~~~js
> (module
>     (table 2 anyfunc)
>     (func $f1 (result i32)
>         i32.const 42)
>     (func $f2 (result i32)
>         i32.const 13)
>     (elem (i32.const 0) $f1 $f2)
>     (type $return_i32 (func (result i32)))
>     (func (export "callByIndex") (param $i i32) (result i32)
>         get_local $i
>         call_indirect (type $return_i32)) // int형만 취급을 하나봄
> ~~~
> ~~~js
> WebAssembly.instantiateStreaming(fetch('test.wasm'))
> .then(resolve => {
>     console.log(resolve.instance.exports.callByIndex(0)); // 42
>     console.log(resolve.instance.exports.callByIndex(1)); // 13
>     console.log(resolve.instance.exports.callByIndex(2)); // error
> })
> ~~~

## Mutating Tables and dynamic linking
> * 자바스크립트는 wasm 코드의 함수 참조에 대한 모든 접근 권한이 있다.
> * Grow(), get(), set() 명령어를 통해 테이블을 변형시킬 수 있고 get_elem(), set_elem()를 사용해서 테이블 자체를 바꿔버릴 수도 있다.
> * 테이블은 변경 가능하기 때문에 **런타임 다이나믹 링크**를 구현하는데 사용할 수 있다.
> * 프로그램이 다이나믹 링크되면 **여러 인스턴스가 동일한 메모리 및 테이블을 공유**할 수 있다.
> * 자바스크립트에서 모든 접근 권한이 있기 때문에 자바스크립트에서 만들고 wasm 코드에 import 해서 사용한다.
> * 아래 코드에서 i32.load와 i32.store는 memory에 저장되어 있는 값을 불러오고, 저장하는 역할을 한다. (data section 이겠지, stack처럼 쓸듯.)
> * 하나의 모듈에는 하나의 테이블이 들어간다.
> * 아래의 코드는 **하나의 테이블이 여러 모듈에 공존**하는 것이다.
> * shared0.wat에서 테이블에 속하는 함수가 정의되었고 export하지 않았다. shared1.wat에서 해당 함수를 콜한다. 테이블은 공유될 수 있기에 가능하다.
> * 테이블 뿐만 아니라 메모리도 공유되었기에 shared1.wat에서 저장한 값이 shared0.wat에서 불러와서 쓸 수 있는 것이다.
> ~~~js
> // shared0.wat
> (module
>     (import "js" "memory" (memory 1))
>     (import "js" "table" (table 1 anyfunc))
>     (elem (i32.const 0) $shared0func)
>     (func $shared0func (result i32)
>         i32.const 0
>         i32.load) // memory 에서 값을 꺼내온다. index는 스택에서 pop하여 0을 가져온다.
> )
> // shared1.wat
> (module
>     (import "js" "memory" (memory 1))
>     (import "js" "table" (table 1 anyfunc))
>     (type $void_to_i32 (func (result i32)))
>     (func (export "doIt") (result i32)
>         i32.const 0
>         i32.const 42
>         i32.store // stack에서 두개를 pop하고 첫번째 0 을 index로 거기에 42를 저장한다. // i32.store (i32.const 0) (i32.const 42) 로 적어도 된다.
>         i32.const 0
>         call_indirect (type $void_to_i32) // call_indirect 함수는 현재 이 모듈에 존재하는 테이블(1개)에 대한 접근을 한다.
> )
> // javascript
> const importObj = {
>     js : {
>         memory : new WebAssembly.Memory({ initial: 1}),
>         table : new WebAssembly.Table({ initial: 1, element: "anyfunc" })
>     }
> };
> Promise.all([
>     WebAssembly.instantiateStreaming(fetch('shared0.wasm'), importObj),
>     WebAssembly.instantiateStreaming(fetch('shared1.wasm'), importObj)
> ]).then(resolve => {
>     console.log(results[1].instance.exports.doIt());
> })
> ~~~