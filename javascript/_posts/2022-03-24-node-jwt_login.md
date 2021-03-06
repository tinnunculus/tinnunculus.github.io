---
layout: post
title: Node- jwt for login
sitemap: false
---

**참고**  
[1] <https://jwt.io/>  
[2] <https://en.wikipedia.org/wiki/JSON_Web_Token>
* * *  

* toc
{:toc}

## login
> * 사용자가 로그인을 완료한 상태이면 다른 페이지로 넘어가거나 다른 요청을 보냈을 시, 서버는 요청을 보낸 사용자가 로그인이 완료된 사용자라는 것을 계속 인식해야만 한다.
> * 즉 사용자는 요청을 보낼 때, 자기 자신이 로그인되었다는 정보를 계속해서 사용자에게 전달해야한다.
> * 대표적으로 쿠키 세션방식과 토큰 방식으로 나누어져있다.

## session 방식
> * 세션방식은 요청의 헤더에 세션 형태로 로그인 정보를 저장하는 것이다.
> * 브라우저에서 세션은 공개되어있기 때문에 반드시 암호화되어야만 하고 중요한 정보를 세션에 저장해서는 안된다.
> * 암호화 되어있기 때문에 세션에 모든 정보를 저장하는 것은 많은 자원을 필요로 한다. (복호화 해야하기 떄문)
> * 그래서 세션에는 유저에 대한 정보를 저장한다기 보다는 데이터베이스의 인덱스만을 세션에 저장해서 디비 검색을 하는 방식으로 진행한다.
> * 그렇기 떄문에 모든 요청 마다 디비 검색을 해야한다는 단점이 있다.

## token 방식
> * 토큰 방식은 세션 방식과 마찬가지로 요청의 헤더에 토큰 형태로 로그인 정보를 저장한다.
> * 토큰 방식은 토큰에 저장된 내용이 공개되기 때문에 세션 방식과 마찬가지로 민감한 정보는 저장해서는 안된다.
> * 하지만 토큰 방식은 인증이 된 사용자에게만 토큰을 전달하기 때문에 토큰을 가지고 있다는 것 만으로 사용자가 인증이 된다.
> * 또한 토큰은 변조되면 변조된 것을 인지할 수 있기 때문에 보안에 안전성이 있다. 그렇다고 해서 다른 사람이 토큰을 갈취해서 해당 사용자로 로그인하는 것을 막을 수는 없다.

## jwt 토큰
> * jwt 토큰은 세 부분으로 나누어진다. Header, Payload, verify signature
> * Header 부분에는 해당 토큰의 타입과 암호화 알고리즘 정보가 들어있다.
> * Payload 부분에는 해당 토큰의 내용이 들어있다.
> * Verify signature 부분에는 토큰의 내용이 암호화되어 들어있어, 서버에서는 암호화된 내용을 복화하여 실제 내용과 비교해서 실제 내용이 변조되어 있는지 확인한다.
> * 세션 방식과 다른 부분은 토큰에 내용이 들어있기 때문에 디비 검색을 할 필요가 없다는 것이지만 패킷에 저장되는 데이터가 많아서 패킷이 무거워지는 단점이 있다.

## jwt 토큰을 통한 로그인 과정
> * jwt 토큰을 통해 로그인하는 과정은 기간이 짧은 Access token과 상대적으로 기간이 긴 Refresh token 두개를 사용한다.
> * 최초의 로그인에서는 Access token과 Refresh token 두개를 발행한다.
> * 그 이후에 클라에서 요청을 보낼때는 Access token을 헤더에 동봉해서 보내고 서버는 해당 토큰을 인증하고 토큰에서 유저 정보를 취득 및 로그인 완료를 한다.
> * Access token의 인증 중에 토큰이 기한 만료가 되면, 클라에서는 Refresh token을 서버에 보내고 토큰 인증 완료되면 서버에서는 access token을 재발행한다.
> * 이렇게 두개의 토큰을 나눠서 인증을 하여 보안을 강화시킨다.
> <p align="center"><img width="550" src="/assets/img/javascript/jwt/1.png"></p>
> <p align="center"><img width="800" src="/assets/img/javascript/jwt/2.png"></p>