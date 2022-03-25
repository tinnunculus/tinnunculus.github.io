---
layout: post
title: Js- class
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/class>  
[2] <https://ko.javascript.info/static-properties-methods>  
[3] <https://ko.javascript.info/private-protected-properties-methods>  
[4] <https://ko.javascript.info/extend-natives>  
[5] <https://ko.javascript.info/mixins>
* * *  

* toc
{:toc}

## 자바스크립트에서 클래스는 사실 함수
> * 자바스크립트에서 클래스는 새롭게 창안한 entitiy가 아니며, 함수의 한 종류이다.
> * 아래의 class 코드를 실행하면 자바스크립트는 User라는 이름을 가진 **함수**를 만든다. **함수의 본문은 constructor 함수에서 가져온다**. constructor가 없다면 body가 비워진 함수로 만들어진다. 그래서 new 연산자를 사용할 수 있는 거구만..
> * sayHi 같은 클래스 내에서 정의한 메서드를 **User.prototype**에 저장한다. field는 prototype에 저장하지 않는다. 메소드만 저장한다고 여겨도 되겟다.
> ~~~js
> class User{
>     constructor(name){this.name = name;}
>     sayHi(){alert(this.name);}
> }
> alert(typeof User); // function
> alert(User == User.prototype.constructor); // true
> alert(User.prototype.sayHi); // alert(this.name)
> alert(Object.getOwnPropertyNames(User.prototype)); // constructor, sayHi
> ~~~
> <p align="center"><img width="550" src="/assets/img/javascript/class/1.png"></p>

## 클래스와 함수의 차이점
> * 클래스와 함수와는 같아 보이지만 약간의 차이가 있다.
> * class로 만든 함수엔 특수 내부 property인 **IsClassConstructor: true** 가 이름표처럼 붙는다.
> * 클래스도 함수지만 일반 함수와 다르게 **new 연산자 없이 사용할 수 없는데**, 그 기반으로 여기는 것이 IsClassConstructor이다.
> * 클래스에 정의된 method는 열거할 수 없다(non-enumerable). 클래스의 prototype property에 추가된 method의 enumerable flag는 false이다.
> ~~~js
> class User {
>   constructor() {}
> }
> User(); // TypeError: Class constructor User cannot be invoked without 'new'
> ~~~

## 클래스 표현식
> * 함수처럼 클래스에도 다양한 표현방식이 있다.
> ~~~js
> // Named Class Expression
> let User = class MyClass { 
>     sayHi() {
>         alert(MyClass); // MyClass라는 이름은 오직 클래스 안에서만 사용할 수 있습니다.
>     }
> };
> new User().sayHi(); // 원하는대로 MyClass의 정의를 보여줍니다.
> alert(MyClass); // ReferenceError: MyClass is not defined, MyClass는 클래스 밖에서 사용할 수 없습니다.
> ~~~
> ~~~js
> // 함수처럼 클래스를 반환할 수도 있다.
> function makeClass(phrase){
>     return class{
>         sayHi(){
>             alert(phrase);
>         }
>     };
> }
> let User = makeClass("Hi~");
> new User().sayHi(); // Hi~
> ~~~

## getter와 setter
> * literal Object처럼 클래스도 getter나 setter를 지원한다.
> * 당연히 method이므로 User.prototype에 정의된다.
> ~~~js
> class User{
>     constructor(name){
>         this.name = name;
>     }
>     get name(){
>         return this._name; // 언더바가 없으면 오류가 난다. 이유는 잘 모르겟다.
>     }
>     set name(value){
>         this._name = value;
>     }
> }
> let user = new User("보라");
> console.log(user.name); // call get name()
> ~~~

## 대괄호를 통한 method 등록
> * literal Object처럼 대괄호를 사용해 계산을 통해 메소드 이름을 등록할 수 있다.
> ~~~js
> class User{
>     ['say' + 'Hi'](){
>         alert("Hello");
>     }
> }
> new User().sayHi();
> ~~~

## 클래스 필드
> * 클래스의 field도 클래스의 property의 한 종류이다.
> * 다른 property랑 다르게 User.prototype에 저장되지 않고 **개별 객체**에 설정된다.
> ~~~js
> class User{
>     name = "종연"; // class field
> }
> let user = new User();
> console.log(user.name); // 종연
> consoler.log(User.prototype.name); // undefined
> ~~~

## 클래스 필드로 바인딩 된 메서드 만들기
> * 자바스크립트에서 this는 **동적**으로 결정된다.
> * 객체 메서드를 여기저기 복사 전달하여 다른 컨택스트에서 호출하게 되면 this는 정의된 객체를 참조하지 않는다.
> * this의 컨택스트를 알 수 없게 되는 문제를 **losing this**라고 한다.
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
> * 위의 코드를 해결하기 위해서는 **setTimeout(function(){button.click();}, 1000);** 을 사용해도 된다.
> * 클래스 필드를 이용해서 해결할 수도 있다.
> * 클래스 필드를 이용해서 **화살표 함수를 만들어주면 그 안에서 this는 항상 객체**로 바인딩된다. (일반함수면 안된다) 이유는 모르겠다. 어쩌면 클래스 필드의 고유한 특징일지도....?
> ~~~js
> class Button{
>     constructor(value){
>         this.value = value;
>     }
>     click = () => { // class field // 화살표 함수로 만들어줘야만 this가 객체로 바인딩된다.
>         console.log(this.value);
>     }
>     click = function(){ console.log(this.value); }
> }
> let button = new Button("안녕하세요.");
> setTimeout(button.click, 1000); // 안녕하세요.
> ~~~
> ~~~js
> function Button(value){
>     this.value = value;
> }
> Button.click = () => { console.log(this.value); }
> Button.click(); // undefined
> ~~~
> ~~~js
> let button = {
>     value: 3,
>     click: () => { console.log(this.vaule); }
> }
> button.click(); // undefined
> ~~~
> ~~~js
> let button2 = {
>     value: 3,
>     click: function() { console.log(this.value); }
> }
> button2.click(); // 3
> ~~~

## 클래스 상속
> * 자바스크립트는 프로토타입 기반의 언어이기 때문에 클래스르 만든다고 하더라도 내부적으로는 프로토타입 기반으로 구현된다.
> * **일반 클래스의 [[prototype]]은 Function**이고 **상속받은 클래스의 [[prototype]]은 부모 클래스(함수)**이다.
> * 아래의 코드는 Animal 클래스를 만들고 Animal 클래스를 상속하는 Rabbit 클래스를 만드는 코드이다.
> * Rabbit 클래스는 constructor 함수를 작성하지 않았다. 그렇다면 Animal 클래스에서 constructor를 불러와서 쓴다.
> ~~~js
> class Animal {
>     constructor(name) { this.speed = 0; this.name = name; }
>     run(speed) { this.speed = speed; }
>     stop() { this.speed = 0; }
> }
> class Rabbit extends Animal {
>   hide() {
>     alert(`${this.name} 이/가 숨었습니다!`);
>   }
> }
> let animal = new Animal("동물");
> let rabbit = new Rabbit("흰 토끼");
> rabbit.run(5);
> rabbit.hide();
> ~~~
> <p align="center"><img width="500" src="/assets/img/javascript/class/2.png"></p>

## 조건마다 다른 클래스르 상속하는 클래스 만들기
> * 자바스크립트의 클래스는 사실상 함수이기 때문에 함수의 리턴문으로 사용할 수 있다
> * 그렇기 때문에 extends class 구문에 class 대신 함수의 리턴문을 넣을 수도 있다.
> * 조건에 따라 다양한 클래스를 넣기 위해서 사용하기에 좋다.
> ~~~js
> // 각기 다른 sayHi 함수를 가진 클래스를 상속하기 위한 것
> function f(phrase){
>     return class{ sayHi(){ console.log(phrase); } }
> }
> class User extends f(phrase) {}
> new User().sayHi();
> ~~~

## constructor 오버라이딩
> * 클래스의 constructor가 비어있다면 디폴트로 constructor 함수를 만들고 부모 클래스의 constructor 함수를 실행시킨다.
> * 상속을 한 클래스에서는 **반드시** 부모 클래스의 constructor 함수를 먼저 실행시켜주어야 한다. 안하면 오류남
> * 자바스크립트에서는 상속받은 클래스의 constructor 함수에서는 특수 내부 property인 [[constructor]]: "derived" 가 존재한다.
> * 일반 클래스 constructor와 상속받은 클래스의 constructor에는 new 연산자와 함께 차이가 난다.
> * 일반 클래스의 constructor가 new와 함께 실행되면, 빈 객체가 만들어지고 this에 이 객체를 할당한다.
> * 상속받은 클래스의 constructor가 new와 함께 실행되면, 빈 객체가 만들어지고 this에는 **아무런 객체를 할당하지 않는다**. 그래서 오류가 걸린다.
> * 따라서 **부모 클래스(일반 클래스)의 constructor를 실행시켜주어 this에 객체를 할당시켜줘야 한다**.
> ~~~js
> class Animal{
>     constructor(name){this.name = name;}
> }
> class Rabbit extends Animal{
>     constructor(name, age){ this.name = name; this.age = age;} // 오류
>     constructor(name, age){ super(name); this.age = age;} // 정상
> }
> rabbit = new Rabbit("jy", 20);
> ~~~

## 클래스 필드의 오버라이딩
> * 클래스는 메소드 뿐만 아니라 내부 **필드**도 오버라이딩 할 수 있다.
> * 필드의 초기화 순서가 일반 클래스와 상속받은 클래스가 다르기 때문에 주의해서 사용할 필요가 있다.
> * 일반 클래스의 필드는 constructor 실행 이전에 초기화 된다.
> * 상속받은 클래스의 필드는 constructor 실행 이후에 초기화 된다.
> * 상속받은 클래스의 constructor에서는 사용하지 말라는 소리
> ~~~js
> class Animal{
>     name = 'animal'
>     constructor(){ alert(this.name); }
>     showfield(){ alert(this.name); }
> }
> class Rabbit extends Animal{
>     name = 'rabbit';
> }
> animal = new Aminal(); // animal
> rabbit = new Rabbit(); // animal
> rabbit.showfield(); // rabbit
> ~~~

## static method in class
> * 클래스의 필드처럼 클래스의 prototype이 아닌 **클래스 함수 자체**에 method를 설정할 수도 있다
> * 이런 method를 **static method**라고 부른다
> * static method는 **클래스의 method를 직접적으로 할당**하는 것과 동일하다
> * static method의 this는 **클래스 함수(객체) 자체**가 된다.
> * 클래스로 만든 **객체에서는 static method에 접근할 수 없다**.
> ~~~js
> class Article{
>     constructor(title, date){ this.title = title; this.date = date; }
>     static compare(articleA, articleB){ return articleA.date - articleB.date; }
> }
> let articles = [
>     new Article("HTML", new Date(2019, 1, 1)),
>     new Article("CSS", new Date(2019, 0, 1)),
>     new Article("JS", new Date(2019, 11, 1))
> ];
> console.log(articles.sort(Article.compare));
> ~~~
> ~~~js
> // 클래스 뿐만 아니라 일반 객체에서도 성립
> function Article(){
>     this.title = arguments[0];
>     this.date = arguments[1];
> }
> Article.compare = function(articleA, articleB){ return articleA.date - articleB.date; }
> let articles = [
>     new Article("HTML", new Date(2019, 1, 1)),
>     new Article("CSS", new Date(2019, 0, 1)),
>     new Article("JS", new Date(2019, 11, 1))
> ];
> console.log(articles.sort(Article.compare));
> ~~~

## static property
> * static method처럼 static property도 존재한다.
> * static method의 특징과 같다.
> * field랑 문법이 비슷해보이니 헷갈리지 않도록 주의하자.
> ~~~js
> class Article{
>     static publisher = "Ilya Kantor";
> }
> console.log(Article.publisher);
> ~~~

## static mothod와 static property의 상속
> * static method와 static property는 일반 mothod와 property처럼 **상속이 가능**하다.
> * 상속 받은 클래스의 **[[prototype]]이 상속한 클래스**를 가리키고 있기 때문이다. (클래스안에 static method, property 들이 있으니 말이다.) 
> ~~~js
> class Animal{
>     static planet = "지구";
> }
> class Rabbit extends Animal {}
> console.log(Rabbit.planet); // "지구"
> ~~~

## 객체를 상속하는 클래스
> * 클래스는 실제로는 함수(객체)이기 때문에 **함수(객체)를 상속**할 수도 있다.
> * 아래의 두 코드들의 차이점을 살펴보자
> ~~~js
> class Rabbit {
>     constructor(name){
>         this.name = name;
>     }
> }
> let rabbit = new Rabbit("Rab");
> console.log(rabbit.hasOwnProperty('name')); // rabbit이 hasOwnProperty method를 가지고 있음에 주의
> console.log(Rabbit.prototype.__proto__ === Object.prototype); // true
> console.log(Rabbit.__proto__ === Object); // false
> console.log(Rabbit.__proto__ === Function.prototype); // true
> ~~~
> ~~~js
> class Rabbit extends Object{
>     constructor(name){
>         super(); // Object() constructor가 실행된다.
>         this.name;
>     }
> }
> let rabbit = new Rabbit("Rab");
> console.log(rabbit.hasOwnProperty('name'));
> console.log(Rabbit.prototype.__proto__ === Object.prototype); // true
> console.log(Rabbit.__proto__ === Object); // true
> ~~~

## private field와 method
> * 자바스크립트에는 private, public 두개의 접근 제한 인터페이스를 제공한다.
> * 디폴트로 사용하는 것이 public이고 이름 앞에 **#을 붙이면 private**로 설정해서 클래스 내에서만 접근할 수 있도록 한다.
> * private 와 public는 상충하지 않기 때문에 두가지 모두 만들 수 있다.
> * **private property는 field를 통해서만 만든다**.
> ~~~js
> class Test{
>     constructor(name){ this.name = name; }
>     #name = "JY"
>     #show_name(){ console.log(this.#name); }
>     show_name(){ console.log(this.name); }
>     show() { this.#show_name(); }
> }
> let test = Test("jongyeon");
> test.show(); // "JY"
> test.show_name(); // "jongyeon"
> ~~~

## 읽기만 가능한 property
> * setter 메소드를 사용하지 않고 private field와 getter 메소드만을 사용한다면 읽기만 가능한 property를 만들 수 있다.
> ~~~js
> class CoffeeMachine{
>     #name = ""
>     constructor(name){ this.#name = name; }
>     get name(){ return this.#name; }
> }
> let coffeemachine = CoffeeMachine("jongyeon");
> console.log(coffeemachine.name); // "jongyeon"
> coffeenachine.name = "test" // error.. no setter
> ~~~

## 내장 클래스 확장하기 및 Symbol.species
> * 자바스크립트에서 기본으로 제공하는 내장 클래스들도 상속이 가능하다
> * 아래의 코드에서 내장 클래스로부터 상속받은 메소드(filter, map)같은 것을 사용할 때 Array 함수의 메소드임에도 Array 객체가 아닌 **PowerArray 객체를 리턴**한다.
> * 이것은 해당 메소드들이 리턴할 때 사용하는 객체 constructor를 **\[Symbol.species]**로 부터 얻는데 디폴트 constructor가 자기자신(PowerArray)이기 때문이다.
> * Symbol.species를 다른 Constructor로 수정하면 다른 결과를 볼 수 있다.
> ~~~js
> class PowerArray extends Array {
>     isEmpty(){ return this.length === 0;}
> }
> let arr = new PowerArray(1, 2, 5, 10, 50);
> console.log(arr.isEmpty()); // false
> let filteredArr = arr.filter(item => item >= 10);
> console.log(filteredArr); // 10, 50
> console.log(filteredArr.isEmpty()); // false ==> filteredArr가 PowerArray의 객체인 것을 알 수 있음.
> ~~~
> ~~~js
> class PowerArray extends Array {
>     isEmpty(){ return this.length === 0;}
>     static get [Symbol.species](){ return Array; }
> }
> let arr = new PowerArray(1, 2, 5, 10, 50);
> console.log(arr.isEmpty()); // false
> let filteredArr = arr.filter(item => item >= 10); // constructor = arr[Symbol.species]
> console.log(filteredArr); // 10, 50
> console.log(filteredArr.isEmpty()); // error : isEmpty() 존재하지 않는다.
> ~~~

## 내장 클래스와 static method 상속
> * 일반 클래스를 상속할 때는 static method와 property 모두 상속이 가능했다.
> * 하지만 **내장 클래스를 상속하는 경우 static method와 property 모두 상속하지 못한다**.
> * 클래스의 **[[prototype]] property가 상속하는 클래스를 가리키지 못하기 때문**이다. 일반 클래스의 상속의 경우 [[prototype]] property는 상속하는 클래스를 가리킨다.
> <p align="center"><img width="500" src="/assets/img/javascript/class/3.png"></p>

## instanceof
> * instanceof 연산자를 사용하면 객체가 특정 클래스에 속하는 지 확인할 수 있다.
> * 해당 클래스뿐만 아니라 parent 클래스 모두 훑어본다.
> * 클래스 뿐만 아니라 생성자 함수에도 사용할 수 있다.
> ~~~js
> function A(){}
> class B extends A {}
> class C extends B {}
> c = new C();
> console.log(c instanceof A); // True 
> ~~~

## instanceof 의 구현
> * 각각의 함수 내부에는 정적 메소드 Symbol.hasInstance가 구현되어 있다. obj instanceof Class 문이 실행될 때, Class\[Symbol.hasInstance](obj) 가 호출된다.
> ~~~js
> class Animal{
>     static [Symbol.hasinstance](obj){ if (obj.canEat) return true; }
> }
> let obj = { canEat : true };
> console.log(obj isinstanceof Animal); // true
> ~~~ 
> * 하지만 대부분의 클래스에는 Symbol.hasinstance가 구현되어 있지 않다. 따라서 [[prototype]] 객체를 확인하는 일반적인 로직을 사용한다.
> * 이 로직은 obj instanceof Class를 Class.prototype.isPrototypeof(obj)와 동일하게 작동한다.
> ~~~js
> // __proto__가 null이 나오거나 Object가 나올때까지 loop를 돌며 true가 존재하면 true를 리턴한다.
> obj.__proto__ === Class.prototype?
> obj.__proto__.__proto__ === Class.prototype?
> obj.__proto__.__proto__ === Class.prototype?
> ~~~
> ~~~js
> function Rabbit(){}
> console.log(new Rabbit() instanceof Rabbit); // true
> Rabbit.prototype = {};
> console.log(new Rabbit() instanceof Rabbit); // false
> ~~~
> ~~~js
> function A() {}
> function B() {}
> A.prototype = B.prototype = {};
> let a = new A();
> console.log(a instanceof B); // true
> ~~~
> <p align="center"><img width="500" src="/assets/img/javascript/class/4.png"></p>

## 메소드 믹스인
> * 특정 클래스를 상속하지 않고 클래스에 있는 method, property들만을 복사해서 사용하고 싶을 때가 있을 수 있다.
> * Object.assign(obj1, obj2); 명령어를 사용하면 obj2에 있는 method, property들을 obj1에 복사해서 사용할 수 있다.
> * 아래 코드에서 super에 주의해서 보자.
> ~~~js
> let sayMixin = {
>     say(phrase) { console.log(phrase); }
> };
> let sayHiMixin = {
>     __proto__: sayMixin,
>     task: "say hello", 
>     sayHi() { super.say(`Hello ${this.name}`) },
>     sayBye() { super.say(`Bye ${this.name}`) }
> };
> class User{
>     constructor(name){ this.name = name; }
> }
> Object.assign(User.prototype, sayHiMixin); // sasyHiMixin에 있는 메소드들을 User.prototype 객체에 복사한다.
> new User("Dude").sayHi();
> ~~~ 
> <p align="center"><img width="500" src="/assets/img/javascript/class/5.png"></p>