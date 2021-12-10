---
layout: post
title: async, await
description: >
  자바스크립트의 async와 await에 대해서 알아보자.
hide_description: true
sitemap: false
---

* toc
{:toc}

## async 함수
> * **function** 앞에 async를 붙이면 해당 함수는 항상 Promise 객체를 반환한다.
> * Promise가 아닌 값(value)를 반환하더라도 이행된(resolved) Promise 객체가 반환되도록 한다.
> * Promise가 resolved 되었다는 것은 excutor 함수가 실행되었다는 소리이다. resolved(value)
> ~~~js
> async function f() {
>     return 1;
> }
> f().then(alert); // 1
> ~~~

## await
> * Promise 객체의 excutor 함수의 결과물은 then, catch, finally 메소드로만 받을 수 있었다. 하지만 이 메소드들은 background에서 비동기적으로 실행된다.
> * await는 excutor 함수의 결과물을 다루는 방법 중 하나로 then 메소드와는 다르게 동기적(결과가 처리될 때까지 기다림)으로 처리한다.
> * await는 **async** 함수 블록 안에서만 동작한다.
> ~~~js
> // then 메소드 이용
> // 1, 3, 2 출력
> async function f(){
>     promise = new Promise(resolve => resolve(2));
>     console.log(1);
>     promise.then(result => console.log(result));
>     console.log(3);
> }
> f();
> // await 메소드 이용
> // 1, 2, 3 출력
> async function g(){
>     promise = new Promise(resolve => resolve(2));
>     console.log(1);
>     prac_await = await promise;
>     console.log(prac_await);
>     console.log(3);
> }
> g();
> ~~~
> 
> ### await의 thenable 객체 받기
> > * then 메소드에서 thenable(then메소드가 있는 Promise가 아닌 객체) 객체를 받을 수 있었던 것 처럼, await에도 thenable 객체를 받을 수 있다.
> > * await는 thenable 객체를 만나면 then 메소드를 실행하고 resolve, reject 함수가 call될 때까지 기다린 후(동기), 그 결과값(result or error)을 받는다.
> > ~~~js
> > function Thenable(num) {
> >     this.num = num;
> >     this.then = function(resolve, reject) {
> >         setTimeout(() => resolve(this.num * 2), 1000);
> >     };
> > }
> > async function f() {
> >     let result = await new Thenable(1);
> >     console.log(result);
> > }
> > f();
> > ~~~
> 
> ### Promise.all
> > * await 구문 또한 Promise.all 메소드를 사용할 수 있다.
> > * 결과물을 배열로 받는다.
> > ~~~js
> > let results = await Promise.all([
> >   fetch(url1),
> >   fetch(url2),
> >   ...
> > ]);
> > ~~~

## 에러 핸들링
> * await는 기존의 then, catch 에러 핸들링과 다르게 함수 블럭 안에 try...catch 구문을 사용할 수 있다.
> ~~~js
> async function practice() {
>     try{
>         let response = await fetch('http://유효하지-않은-주소');
>         let user = await response.json();
>     } catch(err) {
>         alert(err);
>     }
> }
> ~~~
> * try...catch 구문이 아닌 catch 메소드를 사용해서 에러 핸들링을 할 수도 있다.
> ~~~js
> async function f() {
>   let response = await fetch('http://유효하지-않은-url');
> }
> f().catch(alert);
> ~~~

## then 메소드 ==> async, await 코드 예시
> ~~~js
> // then 메소드 
> function showAvatar(githubUser){
>     return new Promise(function(resolve, reject){
>         let img = document.createElement('img');
>         img.src = githubUser.avatar_url;
>         img.className = "promise-avatar-example";
>         document.body.append(img);
>         setTimeout(() => {
>             img.remove();
>             resolve(githubUser);
>         }, 3000);
>     });
> }
> // async, await 메소드
> async function showAvatar(githubUser){
>     let img = document.createElement('img');
>     img.src = githubUser.avatar_url;
>     img.className = "promise-avatar-example";
>     document.body.append(img);
>     await new Promise((resolve, reject) => setTimeout(resolve, 3000));
>     img.remove();
>     return githubUser;
> }
> ~~~