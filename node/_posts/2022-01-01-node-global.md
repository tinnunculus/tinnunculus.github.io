---
layout: post
title: global object and method
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## global
* 노드의 전역객체이다. 브라우저에서는 window이지만 노드에서는 global이다.
* global에 속한 메소드들은 global을 생략해서 적을 수도 있다.
* 노드에는 DOM이 없으므로 document 객체는 노드에서 사용할 수 없다.
~~~js
obj = global.require(파일);
obj = require(파일);
~~~

## Timer
* 타이머 기능을 제공하는 함수는 setTimeout, setInterval, setImmediate이고 모두 이벤트 리스너들이다. 콜백 함수를 가진다.
* 타이머 함수는 global 객체의 메소드들이다.
* setTimeout은 인자로 받는 시간 이후에 이벤트 발생하고 , setInterval은 주어진 시간 마다 이벤트가 반복 발생한다.
* 타이머 메소드들은 모두 자기 자신의 고유한 아이디를 반환한다.
* clearTimeout(아이디), clearInterval(아이디), clearImmediate(아이디) 메소드를 이용해서 해당 아이디를 이용하여 타이머를 취소할 수 있다.
* setTimeout(callback, 0)보다 setImmediate(callback)이 먼저 실행된다.
* 사실 즉시 시행을 하고 싶다면 process.nextTicl(callback)을 사용하는 것이 좋다. 마이크로태스크큐에 저장되기 때문.

## require
* require는 global 객체의 메소드지만 자바스크립트에서 함수는 객체이기 때문에 require도 property와 method들을 가진다.
* require.cache : 그동안 require한 module들을 저장한다. 자기 자신도 포함된다. module에는 exports, parent, children 등등의 property가 존재하며, parent는 자신을 호출한 모듈, children은 자신이 호출한 모듈을 가리킨다. 한번 require한 모듈은 다음번에 require할 때 새로 불러오지 않고 여기서 찾아 간다.
* require.main : root module을 출력한다.
* circular dependency : 서로가 서로를 참조하고 있다면 영원히 서로를 순환적으로 참조하게 되며, 이러한 현상을 참조 의존이라고 부른다. 자바스크립트에서는 이러한 경우 순환 참조가 일어나면 빈 객체를 참조하도록 자체 처리한다.

## process
* 현재 실행중인 노드 프로세스에 대한 정보를 가지고 있다.
### process.env
* process.env는 시스템의 환경 변수를 나타낸다. 
* 서비스의 중요한 키를 저장하는 공간으로 사용된다. 비밀번호 같이 중요한 정보를 코드에 직접 입력하는 것은 보안상 좋지 않다.
~~~js
const secretId = process.env.SECRET_ID; // 환경 변수에서 데이터를 불러 온다.
const secretCode = process.env.SECRET_CODE;
~~~

### process.nextTick(callback)
* 즉시 이벤트가 발생하여 콜백 함수를 실행하는 메소드이다.
* 마이크로테스크큐에 콜백 함수를 올리기 때문에 일반적인 타이머들보다 빨리 실행된다.