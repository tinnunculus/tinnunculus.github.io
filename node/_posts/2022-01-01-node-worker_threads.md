---
layout: post
title: worker threads module
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## worker_threads module
> * 노드는 기본적으로 싱글 스레드 기반이지만 worker_threads 모듈을 사용하면 멀티 스레드를 사용할 수 있다.
> * cpu의 개수만큼 스레드를 생성하는 것이 가장 효율성이 높기 때문에 os.cpus().length 를 이용해서 코어의 개수를 알아내서 그 만큼 스레드를 생성하자.
> * MainThread에서 Worker와 데이터를 주고 받을 때는 항상 이벤트 리스너를 이용해야만 한다. 그렇지 않을 시에는 메인 함수의 전역 함수가 스택을 떠나버려서 더이상 함수를 실행시킬 수 없다. 또한 반복적으로 이벤트를 받아야하기 때문에 promise를 사용할 수는 없다.
> ~~~js
> const { Worker, isMainThread, parentPort } = require('worker_threads');
> const os = require('os');
> if (isMainThread) {
>     const worker = new Worker(__filename);
>     const threads = new Set();
>     const cpu_nums = os.cpus().length;
>     for (let i = 0; i < cpu_nums; i++) {
>         threads.add(new Worker(__filename), {
>             workerData : { start: i + 1 }
>         });
>     for (let worker of threads) {
>         worker.on('message', message => console.log('from worker', message));
>         worker.on('exit', () => {
>             threads.delete(worker);
>             if (threads.size === 0) {
>                 console.log('job done');
>             }
>         });
>     }
> } else {
>     const data = workerData;
>     parentPort.postMessage(data.start + 100);
> }
> ~~~