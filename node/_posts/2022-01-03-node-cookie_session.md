---
layout: post
title: cookie and session
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## 쿠키
* 브라우저는 쿠키라는 것을 이용해서 요청을 보낼 때 마다 지속적으로 서버에 정보를 보낼 수 있다.
* 사용자의 개인정보나 로그인 정보 같은 것을 쿠키에 저장해서 요청마다 서버에 알려주면 새로고침이나 재접속을 해도 로그인이 풀리지 않는다.
* 쿠키는 패킷의 헤더에 저장되며, 단순한 키 : 값 의 쌍이다.
* 쿠키는 유효기간이 존재한다. 브라우저 : 쿠키 의 쌍으로 저장하지 않을까 생각한다.
* 브라우저는 쿠키가 있다면 자동으로 헤더에 동봉해서 보내므로 따로 처리하지 않아도 된다. 그래서 사실상 관리는 서버에서 하는 것 같다.
* 브라우저의 첫번째 요청(주로 접속)에는 쿠키가 존재하지 않고, 응답으로 쿠키를 전송하면 그 다음부터 쿠키를 동봉해서 보낸다.
* 서버에서 Set-Cookie를 사용해서 쿠키를 전송할 때 몇가지 옵션을 넣을 수 있다.
* Expires=날짜 : 만료기한
* Domain=도메인명 : 쿠키가 전송될 도메인을 특정할 수 있다. 기본값은 현재 도메인이다.
* Path=URL : 쿠키가 전송될 URL을 특정할 수 있다. 기본값은 '/'이고 이 경우 모든 URL에서 쿠키를 전송할 수 있다.
* Secure : HTTPS인 경우에만 쿠키가 전송된다.
* HttpOnly : 자바스크립트에서는 쿠키에 접근할 수 없다. 쿠키 조작 방지를 위해 해두는 것이 좋다.
~~~js
http.createServer((req, res) => {
    const cookies = req.headers.cookie; // request header에 동봉된 쿠키 'name=jongyeon' 같이 key=value의 문자열 형식이다.
    const expires = new Date(); 
    expires.setMinutes(expires.getMinutes() + 5); // 유효 시간을 5분으로 설정
    res.writeHead(302, {
        Location: '/', // 이건 뭐징... 쿠키를 저장하는 브라우저 지칭인가??
        'Set-Cookie': `name=${encodeURIComponent('종연')}; Expires=${expires.toGMTString()}; HttpOnly; Path=/` // Path가 쿠키를 저장하는 브라우저 지칭인가??
    })
})
~~~

## 세션
* 쿠키는 브라우저에서 쉽게 접근(노출)할 수 있기 때문에 개인정보 같이 중요한 정보는 쿠키에 직접적으로 넣어두는 것은 적절지 않다.
* 그래서 세션이라는 객체를 만들어서 서버에 저장해두고, 세션의 UID를 쿠키로 저장해서 브라우저에 전송하는 방법으로 해결한다.
* 하지만 세션 ID가 브라우저에서 노출되기 때문에 이 또한 완벽한 방법이라고 할 수는 없다.
~~~js
const session = {};
http.createServer(async (req, res) => {
    ...
    const uniqueInt = Date.now();
    session[uniqueInt] = {
        name,
        expires
    };
    res.writeHed(302, {
        Location: '/',
        'Set-Cookie': `session=${uniqueInt}; ...`
    })
})
~~~