---
layout: post
title: Js- this 
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/arrow-functions>  
[2] <https://ko.javascript.info/object-methods>
* * *  

* toc
{:toc}

## this
> * this는 함수, 메소드 내부에서 쓸 수 있으며 객체를 나타낸다. (다른 언어에서 this는 클래스의 멤버함수 내에서만 쓰이며, 일반적인 함수에서는 쓰지 않는다.)
> * this는 해당 함수를 **호출할 때** 사용된 객체를 뜻한다. 객체를 통해 호출하지 않고 혼자 얼럴뚱땅 호출했으면 전역객체인 Window를 가리킨다.
> * 함수가 정의되거나 인수로 넘겨지거나 인수로 받아질 때가 아닌 함수를 호출할 때 사용된 객체인 점을 명심하자.
> * arr.forEach(func) - func는 전역 객체에 의해 호출된다.
> * setTimeout(func) - func는 전역 객체에 의해 호출된다.
> * new 연산자를 통해 호출한 함수는 새로 생성되는 객체를 가리킨다.
> * 아래 예시는 헷갈리기 쉬운 this의 예시이다.
> ~~~js
> class Button{
>     constructor(value){
>         this.value = value;
>     }
>     click(){
>         console.log(this.value);
>     }
> }
> let button = new Button("안녕하세요.");
> setTimeout(button.click, 1000); // undefined
> ~~~

## 화살표 함수
> * 화살표 함수에는 this와 arguments가 없다. 함수 외부의 this를 의미한다.
> * 아래 코드에서 => 를 쓰지 않고 함수를 전달했으면 this는 전역객체(window)를 나타낼 것이다. 하지만 화살표를 써줌으로써 this를 showList 컨택스트에서의 this를 가져온다.
> ~~~js
> let groub = {
>   title: "study",
>   students: ["종연", "종권", "장원"],
>   showList(){
>     this.students.forEach(student => console.log(this.title)); // => 를 쓰지 않고 함수를 전달했으면 this는 전역객체> (window)를 나타낼 것이다. 하지만 
>   }
> };
> group.showList();
> ~~~
> * 화살표 함수를 사용하면 prototype 객체가 없기 때문에 new 연산자를 사용할 수 없다.
> * this 뿐만 아니라 화살표 함수에는 arguments도 없다.
> * 아래 예시는 화살표 함수가 arguments가 없는 것을 잘 이용한 예시이다.
> ~~~js
> function defer(f, ms){
>   return function(){
>     setTimeout(() => f.apply(this, arguments), ms) // defer함수를 통해 return한 함수를 콜하는 객체의 this와 arguments이다. this는 전역객체(window)를 나타낸다.
>   };
> }
> function sayHi(who) {
>   alert('안녕, ' + who);
> }
> let sayHiDeferred = defer(sayHi, 2000);
> sayHiDeferred("철수"); // 2초 후 "안녕, 철수"가 출력됩니다. arguments = "철수"
> ~~~
> ~~~js
> // 화살표 함수를 사용하지 않고 구현
> function defer(f, ms){
>     return function(...args){
>         let ctx = this;
>         setTimeout(function() {return f.apply(ctx, args);}, ms);  
>     }; 
> }
> ~~~