---
layout: post
title: event listener
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## 이벤트 리스너
* 이벤트는 events 모듈을 통해서 만든다.
* on, once, addListener 메소드를 통해서 이벤트 리스너를 만들 수 있다.
* emit 메소드를 이벤트를 호출할 수도 있다.
* 다른 이벤트 리스너들은 모두 EventEmitter 함수를 상속한다는 것을 알 수 있다.
~~~js
const EventEmitter = require('events');
/////
const event = new EventEmitter();
/////
even.addListener('event1', () => {...});
even.on('event2', () => {...});
even.on('event2', () => {...});
even.once('event3', () => {...});
even.emit('event1');
even.emit('event2'); // event2 리스너 두개가 연달아 콜된다.
even.emit('event3');
even.emit('event3'); // event3는 한번 실행되었기 때문에 실행되지 않는다.
~~~
