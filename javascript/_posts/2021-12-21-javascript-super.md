---
layout: post
title: Js- super
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/class-inheritance>  
* * *  

* toc
{:toc}

## super
> * 상속한 클래스의 **메소드**를 사용하고 싶을 때 super를 이용한다.
> * 혹은 prototype 객체의 **메소드**를 사용하고 싶을 때 super를 이용한다.
> ~~~js
> class Animal{
>     constructor(){}
>     stop(){ console.log("멈춰~!")}
> }
> class Rabbit extends Animal {
>     stop(is_super){
>         if (is_super==true){
>             super.stop();
>         } else {
>             console.log("stop~!");
>         }
>     }
> }
> rabbit = new Rabbit();
> rabbit.stop(is_super=true);
> ~~~
> * 화살표 함수에는 super 객체가 존재하지 않는다.
> ~~~js
> class Rabbit extends Animal{
>     stop(){
>         setTimeout(() => super.stop(), 1000); // Animal.stop() 메소드가 실행
>         setTimeout(function(){ super.stop(); }, 1000); // error: super는 undefined 이다. ps) super는 객체의 메소드, 클래스의 메소드에서만 defined된다.
>     }
> }
> ~~~

## super의 내부
> * super의 역할을 보면 내부의 구현도 어느정도 쉬울 것 같지만 그렇지 않다
> * 엔진은 현재 객체 this를 알기 때문에 this.__proto__.method를 통해 접근할 수 있을 것 같지만 이렇게 하면 오류가 존재한다.
> * 그렇기 때문에 super를 this의 parent라고 생각하면 안된다.
> * 아래 예시를 보면 eat 메소드를 오버라이딩해서 사용하고 있는데 prototype 객체의 메소드를 콜할 때는 원하는 바를 이루기 위해 call(this)를 통해 호출해야 한다. 그렇지 않으면 this는 prototype 객체를 가리킬테니
> * 하지만 아래의 코드를 보면 rabbit.eat() 메소드 무한 루프에 빠지게 된다.
> * 그렇기 때문에 super는 this만으로는 해결할 수 없다.
> ~~~js
> let animal = {
>     name: "동물",
>     eat() {
>         console.log(`${this.name}이 먹이를 먹는다.`);
>     }
> };
> let rabbit = {
>     __proto__: animal,
>     name: "토끼",
>     eat(){
>         this.__proto__.eat.call(this);
>     }
> };
> let longEar = {
>     __proto__: rabbit,
>     eat() {
>         this.__proto__.eat.call(this);
>     }
> };
> longEar.eat(); // RangeError: Maximum call stack size exceeded
> ~~~

## [[HomeObject]]
> * super는 this를 통해 구현할 수 없다.
> * [[HomeObject]]라는 **함수 전용** 특수 property를 이용하면 해결할 수 있다.
> * [[HomeObject]]는 this처럼 해당 객체가 저장된다.
> * super는 [[Homeobject]]를 이용해서 parent 객체를 찾고 처음 호출할 당시 자신의 객체(this)를 유지한다.
> ~~~js
> let animal = {
>     name: "동물",
>     eat() { // eat.[[HomeObject]] = animal
>         console.log(`${this.name}이 먹이를 먹는다.`); // this는 longEar로 들어간다.
>     }
> };
> let rabbit = {
>     __proto__: animal,
>     name: "토끼",
>     eat(){ // eat.[[HomeObject]] = rabbit
>         super.eat();
>     }
> };
> let longEar = {
>     __proto__: rabbit,
>     eat() { // eat.[[HomeObject]] = longEar
>         super.eat();
>     }
> };
> longEar.eat(); // this는 longEar가 들어간다. super가 HomeObject를 통해 그렇게 되도록 유지한다.
> ~~~
> * [[HomeObject]] property는 **변경할 수 없다**. 그렇기 때문에 함수(객체)가 생성될 때 **고정**되어 정해진다.
> * 하지만 자바스크립트에서 **함수나 객체는 자유롭게 복사되거나 바인딩 객체가 변할 수도 있다**.
> * [[HomeObject]] property는 super 내부에서만 유효하다. 하지만 복사와 바인딩을 super와 함께사용한다면 문제가 될 수 있다.
> * 아래의 코드를 보면 tree.sayHi() 메소드는 rabbit.sayHi() 메소드를 **복사**해 쓴다.
> * 하지만 복사한 메소드는 rabbit에 의해 생성된 함수이므로 함수의 [[HomeObject]] property는 rabbit이다.
> * super는 [[HomeObject]] 객체를 통해 super를 찾기 때문에 animal 객체를 super로 인지한다.
> * 이것이 this와 super의 다른 접근이라고 볼 수 있겠다.
> ~~~js
> let animal = {
>     sayHi(){ console.log("animal"); }
> };
> let rabbit = {
>     __proto__: animal,
>     sayHi(){ super.sayHi(); }
> };
> let plant = {
>     sayHi(){ console.log("plant"); }
> };
> let tree = {
>     __proto__: plant,
>     sayHi(): rabbit.sayHi
> };
> tree.sayHi(); // animal
> ~~~
> * [[HomeObject]] 메소드에서만 정의가 된다. 하지만 method()의 형태로 정의해야만 한다. method: function() 형태로 정의해서는 안된다.
> * method: function(){...} 내부에서 super를 사용할 경우 SyntaxError가 발생한다.

## 화살표 함수
* 화살표 함수는 function을 __proto__를 가지긴 하지만, **prototype property를 가지지 않는다**.
~~~js
let test = () => {};
console.log(test.prototype); // undefined
~~~