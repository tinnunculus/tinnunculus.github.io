---
layout: post
title: this -writting-
sitemap: false
---

**참고**  
[1] 조현영, Node.js 교과서, 길벗(2021), p  
[2] <https://ko.javascript.info/function-prototype>  
* * *  

* toc
{:toc}

* 객체의 메소드 안의 this는 해당 메소드를 호출한 객체이다.
* 아래의 코드에서는 animal 안에 sleep 메소드에 this가 있지만 sleep 메소드를 rabbit 객체를 통해 접근하면 해당 this는 animal이 아니라 rabbit이 된다.
~~~js
let animal = {
    walk() {
        if (!this.isSleeping) {
            alert('동물이 깨어있습니다.');
        }
    },
    sleep() {
        this.isSleeping = true;
    }
};
let rabbit = {
    name: "white rabbit",
    __proto__: animal
};
rabbit.sleep();
alert(rabbit.isSleeping); // true
alert(animal.isSleeping); // undefined
~~~


* 아래의 코드가 어떻게 실행되는지 알아보쟈
~~~js
Function.prototype.defer = function(ms) {
  let f = this;
  return function(...args) {
    setTimeout(() => f.apply(this, args), ms); // f와 this의 차이점.
  }
};

// 확인해 보세요.
function f(a, b) {
  alert( a + b );
}

f.defer(1000)(1, 2); // 1초 후 3 출력
~~~