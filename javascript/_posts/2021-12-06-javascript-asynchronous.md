---
layout: post
title: Background environment of asynchronous function -writting-
description: >
  자바스크립트의 비동기의 실행원리에 대해서 알아보자.
hide_description: true
sitemap: false
---

* toc
{:toc}

* 아래의 코드가 어떻게 실행되는지 알아보쟈
~~~js
Function.prototype.defer = function(ms) {
  let f = this;
  return function(...args) {
    setTimeout(() => f.apply(this, args), ms);
  }
};

// 확인해 보세요.
function f(a, b) {
  alert( a + b );
}

f.defer(1000)(1, 2); // 1초 후 3 출력
~~~