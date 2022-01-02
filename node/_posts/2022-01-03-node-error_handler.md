---
layout: post
title: error handler
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## try, throw, catch
* 노드는 메인 스레드가 하나 뿐이라, 메인 스레드가 멈추면 시스템 자체가 멈춘다.
* 에러가 발생하더라도 치명적인 에러가 아니라면 로그로 기록을 해놓고 메인 스레드는 작업을 계속 진행하도록 해야한다.
* try, throw, catch 구문을 이용해서 항상 에러처리를 해주는 것이 좋다.
* 콜백 함수 밖, 이벤트 리스너에서는 try, catch 구문으로 에러처리가 되지 않으니 콜백 함수 내에서 try, catch 구문을 사용하는 것이 좋다.
~~~js
setInterval(() => {
    try {
        throw new Error('서버를 고장내!!');
    } catch (err) {
        consol.error(err);
    }
}, 1000); // 이벤트 리스너 밖에서는 콜백 함수애서 발생하는 에러를 잡을 수 없다.
~~~

## uncaughtException 
* uncaughtException은 에러를 catch 구문을 통해서 잡지 못하면 프로세스가 발생시키는 이벤트이다.
* 프로세스 객체를 통해서 이벤트를 받을 수 있다.
* 노드는 uncaughtException 이벤트 발생 이후 작업이 제대로 진행된다는 것을 보장시키지 못한다.
* uncaughtException 이벤트는 오류를 기록하는 정도로 사용하는 것이 좋다.
~~~js
process.on('uncaughtException', (err) => {
    console.error(err);
});
~~~