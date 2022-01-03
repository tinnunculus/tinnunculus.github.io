---
layout: post
title: https and http2
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## https
* http에 SSL 암호화를 추가한 것을 https라고 한다.
* 요청과 응답에 오가는 데이터를 암호화한다.
* 전문 인증기관에서 인증서와 비밀 키를 발급받아야 한다. 인증서는 .pem 포멧, 인증키는 .key 포멧이다.
~~~js
const https = require('https');
https.createServer({
    cert: fs.readFileSync(도메인 인증서 경로),
    key: fs.readFileSync(도메인 비밀키 경로),
    ca: [
        fs.readFileSync(상위 인증서 경로),
        fs.readFileSync(상위 비밀키 경로)
    ],
}, (req, res) => {...});
~~~

## http2
* http2는 https와 더불어 새로운 파이프라인을 적요해서 기존의 http 보다 확연히 빠른 속도이다.
* 코드는 https와 거의 동일하다. createServer 메소드를 createSecureServer 메소드로 바꾸면 된다.
<p align="center"><img width="550" src="/assets/img/node/https_http2/1.png"></p>