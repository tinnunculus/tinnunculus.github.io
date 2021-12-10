---
layout: post
title: Background environment of asynchronous function -writting-
sitemap: false
---

**참고**  
[1]. 조현영, Node.js 교과서, 길벗(2021), p  
* * *  

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