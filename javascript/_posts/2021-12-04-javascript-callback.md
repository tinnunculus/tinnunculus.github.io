---
layout: post
title: Js- callback
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/callbacks>  
* * *  

* toc
{:toc}

## 자바스크립트에서의 비동기
> * 자바스크립트에서는 여러 함수를 비동기(aynchronous) 동작을 스케줄링할 수 있다.
> * 비동기라고 해도 멀티프로세스나 멀티스레드 프로그래밍을 하지 않는 이상 기본적으로 싱글 스레드 위에서 돌아가기 때문에 동시에 진행되지는 않는다.
> * 하지만 이러한 비동기 시스템 때문에 발생하는 문제들도 있다.
> 
> ### 비동기 함수 사용시 주의할 점.
> > * 아래 코드는 document 객체를 이용해서 HTML문서에 스크립트 태그를 **동적**으로 삽입하는 코드이다.
> > * DOM API를 통해 로딩하는 것은 비동기로 실행된다. 즉, 함수를 콜하면 바로 실행되는 것이 아닌 백그라운드로 이동 후 실행된다.
> > * 만약에 new_function 함수가 test.js 파일에 있는 함수라면, 스크립트가 로딩되기 전에 new_function 코드가 실행될 수 있어 에러가 발생한다.
> > * 위의 문제를 해결하기 위해서는 자바스크립트의 스크립트 소스가 전부 로딩된 이후에 new_function 코드가 실행되어야 한다.
> > ~~~js
> > function loadScript(src){
> >     let script = document.createElement('script'); // document DOM API를 이용해서 DOM 트리를 만듬, script 요소(태그)를 만듬.  
> >     script.src = src; // script 요소에 src 속성 대입.  
> >     document.head.append(script); // DOM head에 새로만든 요소 추가. // 비동기
> > }
> > loadScript('test.js');
> > new_function();
> > ~~~

## 콜백 함수
> * 아래와 같은 방식을 콜백 기반의 비동기 프로그래밍이라고 한다.
> * **onload** 함수는 스크립트 로딩이 된 이후에 발생하는 이벤트에 의해 실행되는 이벤트 함수이다.
> * 때문에 스크립트가 로딩이 된 이후에 newFunction() 이 실행된다.
> * 이런식으로 어떠한 이벤트가 발생 후 실행되는 함수들을 callback 함수라고 한다.
> ~~~js
> function loadScript(src, callback){
>     let sript = document.createElement('script');
>     script.src = src;
>     script.onload = () => callback(script); // 스크립트 로딩 후 자동으로 실행된다.
>     document.head.append(script);
> }
> loadScript('test.js', function(){ newFunction(); });
> ~~~
> 
> ### 콜백 중첩
> > * 위의 예시에서 로딩하고 싶은 콜백 함수가 두개 이상이고 순차적으로 로딩하고 싶다면 콜백 함수 안에 콜백 함수를 실행하면 될 것이다.
> > * 아래의 코드는 1.js 파일이 로드된 이후에 2.js 파일이 로드될 것이다.
> > ~~~js
> > loadScript('1.js', function(){
> >     loadScript('2.js', function(){
> >     }
> > }
> > ~~~
> 
> ### 콜백 함수를 이용한 에러 핸들링
> > * 콜백 함수를 이용해서 에러 핸들링을 할 수 있다.
> > * 스크립트 로딩에 성공하면 callback(null, script)을, 실패하면 callback(error)함수를 콜한다.
> > * 이렇게 콜백을 이용하여 오류를 처리하는 방식을 error first callback이라고 부른다.
> > * callback 함수의 첫번째 인자는 에러를 위해 남겨두고, 에러가 발생하면 에러 인자를 받고 안 발생하면 null 값을 받는다.
> > * 두번째 인자는 에러가 발생하지 않을 때를 대비하여 남겨둔다.
> > ~~~js
> > function loadScript(src, callback){
> >     let script = document.createElement('script');
> >     script.src = src;
> >     script.onload = () => callback(null, script); // 스크립트가 정상적으로 로딩되면 발생하는 이벤트 함수
> >     script.onerror = () => callback(new Error("에러발생")); // 에러가 발생하면 발생하는 이벤트 함수
> >     document.head.append(script);
> > }
> > loadScript('1.js', function(error, script){
> >     if (error){
> >         // 에러 처리
> >     } else {
> >         // 성공적인 스크립트 로딩
> >     }
> > });
> > ~~~  
> > #### 콜백 지옥
> > > * 콜백 함수는 여러 장점이 있지만 중첩이 여러번되다 보면 코드의 가독성이 나빠지기 때문에 후에 중첩이 많다면 후에 나올 프로미스 방식을 채택한다.
