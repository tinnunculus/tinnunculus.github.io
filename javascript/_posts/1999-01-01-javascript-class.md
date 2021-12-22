---
layout: post
title: class
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/class>  
[2] <https://ko.javascript.info/static-properties-methods>
* * *  

## 자바스크립트에서 클래스는 사실 함수
* 자바스크립트에서 클래스는 새롭게 창안한 entitiy가 아니며, 함수의 한 종류이다.
* 아래의 class 코드를 실행하면 자바스크립트는 User라는 이름을 가진 함수를 만든다. 함수의 본문은 constructor 함수에서 가져온다. constructor가 없다면 body가 비워진 함수로 만들어진다.
* sayHi 같은 클래스 내에서 정의한 메서드를 User.prototype에 저장한다.
~~~js
class User{
    constructor(name){this.name = name;}
    sayHi(){alert(this.name);}
}
alert(typeof User); // function
alert(User == User.prtotype.constructor); // true
alert(User.prototype.sayHi); // aert(this.name)
alert(Object.getOwnPropertyNames(User.prototype)); // constructor, sayHi
~~~
<p align="center"><img width="550" src="/assets/img/javascript/class/1.png"></p>

## 클래스와 함수의 차이점
* 자바스크립트에서 클래스는 그동안 사용했던 프로토타입 기반으로 상속해서 사용하던 것의 코드 편의성을 위하여 나타난 것으로 생각하기 쉽지만 아니다.
* class로 만든 함수엔 특수 내부 property인 IsClassConstructor: true 가 이름표처럼 붙는다.
* 클래스도 함수지만 일반 함수와 다르게 new연산자 없이 사용할 수 없는데, 그 기반으로 여기는 것이 IsClassConstructor이다.
* 클래스에 정의된 method는 열거할 수 없다(non-enumerable). 클래스의 prototype property에 추가된 method의 enumerable flag는 false이다.
~~~js
class User {
  constructor() {}
}

alert(typeof User); // User의 타입은 함수이긴 하지만 그냥 호출할 수 없습니다.
User(); // TypeError: Class constructor User cannot be invoked without 'new'
~~~

## 클래스 표현식
* 함수처럼 클래스에도 다양한 표현방식이 있다.
~~~js
// Named Class Expression
let User = class MyClass { 
    sayHi() {
        alert(MyClass); // MyClass라는 이름은 오직 클래스 안에서만 사용할 수 있습니다.
    }
};
new User().sayHi(); // 원하는대로 MyClass의 정의를 보여줍니다.(new User()를 통해서 객체생성 후 sayHi 메소드 실행)
alert(MyClass); // ReferenceError: MyClass is not defined, MyClass는 클래스 밖에서 사용할 수 없습니다.
~~~
~~~js
// 함수처럼 클래스를 반환할 수도 있다.
function makeClass(phrase){
    return class{
        sayHi(){
            alert(phrase);
        }
    };
}
let User = makeClass("Hi~");
new User().sayHi(); // Hi~
~~~

## getter와 setter
* literal Object처럼 클래스도 getter나 setter를 지원한다.
* 당연히 method이므로 User.prototype에 정의된다.
~~~js
class User{
    constructor(name){
        this.name = name;
    }
    get name(){
        return this._name; // 언더바가 없으면 오류가 난다. 이유는 잘 모르겟다.
    }
    set name(value){
        this._name = value;
    }
}
let user = new User("보라");
console.log(user.name);
user = new User("사랑");
~~~

## 대괄호를 통한 method 등록
* literal Object처럼 대괄호를 사용해 계산을 통해 메소드 이름을 등록할 수 있다.
~~~js
class User{
    ['say' + 'Hi'](){
        alert("Hello");
    }
}
new User().sayHi();
~~~

## 클래스 필드
* 클래스의 property의 한 종류이다.
* 다른 property랑 다르게 User.prototype에 저장되지 않고 개별 객체에 property가 설정된다.
~~~js
class User{
    name = "종연"; // class field
}
let user = new User();
console.log(user.name); // 종연
consoler.log(User.prototype.name); // undefined
~~~

## 클래스 필드로 바인딩 된 메서드 만들기
* 자바스크립트에서 this는 동적으로 결정된다.
* 객체 메서드를 여기저기 전달해 다른 컨택스트 에서 호출하게 되면 this는 정의된 객체를 참조하지 않는다.
* this의 컨택스트를 알 수 없게 되는 문제를 losing this라고 한다.
~~~js
class Button{
    constructor(value){
        this.value = value;
    }
    click(){
        console.log(this.value);
    }
}
let button = new Button("안녕하세요.");
setTimeout(button.click, 1000); // undefined
~~~
* 위의 코드를 해결하기 위해서는 setTimeout(() => button.click(), 1000); 을 사용해도 된다.
* 클래스 필드를 이용해서 해결할 수도 있다.
* 클래스 필드로 만든 함수는 객체마다 독립적인 함수로 만들어주고 이 함수의 this를 해당 객체에 바인딩시켜준다.
* 사용자는 button.click을 아무 곳에나 전달할 수도 있고, this엔 항상 의도한 값이 들어간다.
~~~js
class Button{
    constructor(value){
        this.value = value;
    }
    click = () => { // class field
        console.log(this.value);
    }
}
let button = new Button("안녕하세요.");
setTimeout(button.click, 1000); // 안녕하세요.
~~~

## 클래스 상속
* 자바스크립트는 프로토타입 기반의 언어이기 때문에 클래스르 만든다고 하더라도 내부적으로는 프로토타입 기반으로 구현된다.
* 아래의 코드는 Animal 클래스를 만들고 Animal 클래스를 상속하는 Rabbit 클래스를 만드는 코드이다.
* Rabbit 클래스는 constructor 함수를 작성하지 않았다. 그렇다면 Animal 클래스에서 constructor를 불러와서 쓴다.
~~~js
class Animal {
    constructor(name) {
        this.speed = 0;
        this.name = name;
    }
    run(speed) {
        this.speed = speed;
        alert(`${this.name} 은/는 속도 ${this.speed}로 달립니다.`);
    }
    stop() {
        this.speed = 0;
        alert(`${this.name} 이/가 멈췄습니다.`);
    }
}
class Rabbit extends Animal {
  hide() {
    alert(`${this.name} 이/가 숨었습니다!`);
  }
}
let animal = new Animal("동물");
let rabbit = new Rabbit("흰 토끼");
rabbit.run(5); // 흰 토끼 은/는 속도 5로 달립니다.
rabbit.hide(); // 흰 토끼 이/가 숨었습니다!
~~~
<p align="center"><img width="500" src="/assets/img/javascript/class/2.png"></p>

## 조건마다 다른 클래스르 상속하는 클래스 만들기
* 자바스크립트의 클래스는 사실상 함수이기 때문에 함수의 리턴문으로 사용할 수 있다
* 그렇기 때문에 extends class 구문에 class 대신 함수의 리턴문을 넣을 수도 있다.
* 조건에 따라 다양한 클래스를 넣기 위해서 사용하기에 좋다.
~~~js
// 각기 다른 sayHi 함수를 가진 클래스를 상속하기 위한 것
function f(phrase){
    return class{
        sayHi(){ console.log(phrase); }
    }
}
class User extends f(phrase) {}
new User().sayHi();
~~~

## constructor 오버라이딩
* 클래스의 constructor가 비어있다면 디폴트로 constructor 함수를 만들고 부모 클래스의 constructor 함수를 실행시킨다.
* 상속을 한 클래스에서는 **반드시** 부모 클래스의 constructor 함수를 먼저 실행시켜주어야 한다. 안하면 오류남
* 자바스크립트에서는 상속한 클래스의 constructor 함수에서는 상속하지 않은 클래스의 constructor 함수와 다른 특수 내부 property인 [[constructor]]: "derived" 가 존재한다.
* 일반 클래스 constructor와 상속 클래스의 constructor에는 new 연산자와 함께 차이가 난다.
* 일반 클래스의 constructor가 new와 함께 실행되면, 빈 객체가 만들어지고 this에 이 객체를 할당한다.
* 상속 클래스의 constructor가 new와 함께 실행되면, 빈 객체가 만들어지고 this에는 아무런 객체를 할당하지 않는다.
* 따라서 부모 클래스(일반 클래스)의 constructor를 실행시켜주어 this에 객체를 할당시켜줘야 한다.
~~~js
class Animal{
    constructor(name){this.name = name;}
}
class Rabbit extends Animal{
    constructor(name, age){ this.name = name; this.age = age;} // 오류
    constructor(name, age){ super(name); this.age = age;} // 정상
}
rabbit = new Rabbit("jy", 20);
~~~

## 클래스 필드의 오버라이딩
* 클래스는 메소드 뿐만 아니라 내부 필드도 오버라이디 할 수 있다.
* 필드의 초기화 순서가 일반 클래스와 상속한 클래스가 다르기 때문에 주의해서 사용할 필요가 있다.
* 일반 클래스의 필드는 constructor 실행 이전에 초기화 된다.
* 상속한 클래스의 필드는 constructor 실행 이후에 초기화 된다.
~~~js
class Animal{
    name = 'animal'
    constructor(){
        alert(this.name);
    }
    showfield(){
        alert(this.name);
    }
}
class Rabbit extends Animal{
    name = 'rabbit';
}
animal = new Aminal(); // animal
rabbit = new Rabbit(); // animal
rabbit.showfield(); // rabbit
~~~

## static method in class
* 클래스의 필드처럼 클래스의 prototype이 아닌 클래스 함수 자체에 method를 설정할 수도 있다
* 이런 method를 static method라고 부른다
* static이라 부르는 이유는 객체를 생성하지 않고도 클래스 함수 자체만으로 접근할 수 있기 때문이 아닐까
* static method는 클래스의 method를 직접적으로 할당하는 것과 동일하다
* static method의 this는 클래스 함수(객체) 자체가 된다.
~~~js
class Article{
    constructor(title, date){
        this.title = title;
        this.date = date;
    }
    static compare(articleA, articleB){
        return articleA.date - articleB.date;
    }
}
let articles = [
    new Article("HTML", new Date(2019, 1, 1)),
    new Article("CSS", new Date(2019, 0, 1)),
    new Article("JS", new Date(2019, 11, 1))
];
console.log(articles.sort(Article.compare)); // 클래스 함수(객체) 자체가 this가 되어 날라가는 모습
////////////////////////////////////////
//// 클래스 뿐만 아니라 일반 객체에서도 성립 ////
function Article(){
    this.title = arguments[0];
    this.date = arguments[1];
}
Article.compare = function(articleA, articleB){
    return articleA.date - articleB.date;
}
let articles = [
    new Article("HTML", new Date(2019, 1, 1)),
    new Article("CSS", new Date(2019, 0, 1)),
    new Article("JS", new Date(2019, 11, 1))
];
console.log(articles.sort(Article.compare)); // 클래스 함수(객체) 자체가 this가 되어 날라가는 모습
~~~