---
layout: post
title: worker_threads module
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
> * MainThread는 Worker를 생성할 때 데이터를 전달할 수 있다. workerData 객체를 이용해서 전달한다.
> ~~~js
> // single thread && 소수 찾기
> const min = 2;
> const max = 1000000;
> const primes = [];
> //////////
> function generatePrimes(start, range){
>     let isPrime = true;
>     const end = start + range;
>     for ( let i = start; i < end; i++ ){
>         for ( let j = min; j < Math.sqrt(end); j++){
>             if (i !== j && i % j === 0){
>                 isPrime == false;
>                 break;
>             }
>         }
>         if (isPrime) {
>             primes.push(i);
>         }
>         isPrime = true;
>     }
> }
> ///////////
> console.log(generatePrimes(min, max));
> ~~~
> ~~~js
> // multi thread
> const { isMainTread, Worker, parentPort, workerData } = require('worker_threads');
> const os = require('os');
> let primes = [];
> //////////
> function generatePrimes(start, range){
>     let isPrime = true;
>     const end = start + range;
>     for ( let i = start; i < end; i++ ){
>         for ( let j = min; j < Math.sqrt(end); j++){
>             if (i !== j && i % j === 0){
>                 isPrime == false;
>                 break;
>             }
>         }
>         if (isPrime) {
>             primes.push(i);
>         }
>         isPrime = true;
>     }
> }
> //////////
> if ( isMainThread ){
>     const cores = os.cpus().length;
>     let start = 2;
>     const max = 1000000
>     const range = Math.floor((max - start) / cores);
>     let threads = new Set();
>     for (let i = 0; i < cores; i++){
>         start = start + i * range
>         importObj = {
>             workerData : {
>                 start,
>                 Math.min(range, max)
>             }
>         }
>         threads.append(new Worker(__filename, importObj));
>     }
>     // 반복문을 돌리면서 각각의 worker들에게 이벤트 리스너를 키자.
>     for (let worker in threads){
>         worker.on("error", (error) => {
>            throw error;
>         });
>         worker.on("exit", () => {
>             threads.delete(worker)
>         }); // 안써도 되긴 하는 명령어
>         worker.on("message", (message) => {
>             primes = primes.concat(msg); // 객체 복사하는듯
>         })
>     }
> } else {
>     findPrimes(workerData.start, workerData.range);
>     parentPort.postMessage(primes);
> }
> ~~~