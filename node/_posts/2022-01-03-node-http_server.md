---
layout: post
title: http server
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## http server
* 서버와 클라이언트간에 데이터를 주고 받을 때, 주고 받는 데이터의 구조가 http 양식을 따르는 서버를 http 서버라고 한다.
* 노드에서는 http server를 위한 라이브러리를 제공한다.
* http.createServer(callback)를 통해 클라이언트의 요청에 응답하는 이벤트 리스너를 만들고, 해당 이벤트 리스너를 특정 port 번호에 연결을 하여 http 서버를 만든다.
* createServer 이벤트 리스너는 계속 요청을 받는다.
* 클라이언트에서 서버의 프로그램에 접근하기 위해서는 port 번호를 필요로 하기 때문에 port 번호를 반드시 설정해야만 한다. http는 80, https는 443 port 번호를 사용하면 url에서 port 번호를 생략하고 접속할 수 있다.
* 한 머신에서 여러 서버를 만들고 실행할 수 있다.
* 요청이 성공이든 실패이든 어떤 내용이든 간에 응답은 반드시 보내야만 한다.
* http state code : 2xx (성공), 3xx (다른 페이지로 이동), 4xx (요청 자체에 오류), 5xx (서버 오류)
~~~js
const http = require('http');
/////
const server1 = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); // 응답의 헤더 부분을 들어가는 데이터에 관한 정보
    res.write('<h1>Hello Node</h1>');
    res.end('<p>Hello Server1</p>'); // 응답의 끝
});
const server2 = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.write('<h1>Hello Node</h1>');
    res.end('<p>Hello Server2</p>');
});
server1.listen(8080); // 8080 port 번호를 설정.
server2.listen(8081); // 8081 port 번호를 설정.
server1.on('listening', () => { console.log('server1 작동'); });
server2.on('listening', () => { console.log('server2 작동'); });
~~~
* 아래의 코드는 fs 모듈을 통해 html 파일을 읽고 클라이언트에 응답으로 전송하는 코드이다.
~~~js
const http = require('http');
const fs = require('fs').promises;
/////
server = http.createServer( async (req, res) => {
    try {
        data = await fs.readFile('./test.html');
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(data); // 버퍼에 들어가는 데이터 형식 그대로 전송한다.
    } catch (err) {
        console.error(err);
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end(err.message);
    }
});
server.listen(8080);
server.on('listening', () => { console.log('서버 작동 시작'); });
~~~

## axios를 이용한 http 요청
* axios.get : 서버의 자원을 가져올 때 사용한다. 요청의 body에는 데이터를 넣지 않고, 데이터를 서버로 보내야 한다면 주소의 querystring을 이용한다.
* axios.post : 서버에 자원을 새로 등록할 때 사용한다. 요청의 body에는 등록할 데이터를 넣는다.
* axios.put : 서버의 자원을 요청에 들어 있는 자원으로 치환한다. 요청의 body에는 치환할 데이터를 넣는다.
* axios.patch : 서버의 자원을 일부만 수정할 때 사용한다. 요청의 body에는 수정할 데이터를 넣는다.
* axios.delete : 서버의 자원을 삭제할 때 이용한다. 요청의 body에는 데이터를 넣지 않는다.
* axios를 이용한 요청은 url 주소만 봐서는 무슨 말인지 알 수 없다. 요청의 종류가 무엇인지 따로 확인해야만 한다.
* 클라이언트에서 axios 요청을 보내면 서버에서는 req.method를 통해 어떤 요청인지 알아낼 수 있다. (GET, POST, PUT, PATCH, DELETE)
* 책의 예제에서는 클라이언트가 html 파일만 요청하면 html 파일만을 주었고 자바스크립트 파일은 주지 않았다. 그럼 자바스크립트 파일은 어떻게 실행되는 걸까...?
* post나 put 같이 본문에 데이터를 동봉해서 보내는 경우 스트림으로 받기 때문에 서버에서는 req.on('data', (data)=>{}) 을 통해 데이터를 받고 JSON 형식의 문자열로 데이터를 받는 듯 하다. html 데이터를 전달할 때는 Buffer를 전달했는데 데이터가 올 때는 해석?되어서 문자열로 오니 신기하다. 데이터가 다 전달 되면 req.on('end', ()=>{}) 이벤트가 발생한다.
* 이러한 axios를 이용한 요청에 응답하는 이벤트 리스너를 라우트라고 한다.