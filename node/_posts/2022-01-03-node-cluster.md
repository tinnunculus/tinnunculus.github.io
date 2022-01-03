---
layout: post
title: cluster
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## cluster
* 노드에서 사용할 수 있는 멀티 프로세스 기능이다.
* 멀티 스레드와 프로그래밍이 비슷하지만 멀티 프로세스 기능이기 때문에 메모리를 공유하지 못한다. 데이터를 전송이 어려움??
* child_process 처럼 다른 종류의 프로세스를 실행하는 것이 아닌 자기 자신을 fork해서 실행하는 방식의 멀티 프로세스이다.
* 여러 서버를 열어서 모두 같은 포트 번호에서 대기하도록 설정할 수도 있다. 이렇게 하면 한 프로세스에서 오류가 발생해도 다른 프로세스에서 진행하면 되기 때문에 안정성이 좋음.
* 워커마다 다른 일을 하기에 좋음. 하지만 모두 같은 포트 번호를 할당하면 일단 데이터를 모두에게 받는다는 것은 자원 비효율적이지 않은가??,,
* 마스터 프로세스에서 데이터를 받고 일을 워커에게 나눠주는게 나을둣...?
~~~js
const cluster = require('cluster');
const http = require('http');
const numCPUs = require('os').cpus().length
/////
if (cluster.isMaster){
    for (let i=0; i<numCPUs; i++) {
        cluster.fork();
    }
    cluster.on('exit', (worker, code, signal) => { 
        console.log('${worker.process.pid}번 워커가 종료되었다.');
        if (code == 0){ 
            cluster.fork(); 
        }
    });
} else {
    http.createServer((req, res) => {
        ...
    }).listen(8080);
    console.log(`${process.pid}번 워커 실행`);
}
~~~