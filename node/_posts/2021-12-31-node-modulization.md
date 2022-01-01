---
layout: post
title: module
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## module
> * 자바스크립트에서 모듈은 자바스크립트 파일을 기준으로 한다.
> * 다른 자바스크립트에서 사용할 수 있도록 해당 파일을 모듈화 해서 transfer한다.
> * id : 자기 자신을 가리키는 주소
> * exports : 내보낼 객체, 함수, 변수
> * filename : 자신의 파일 주소
> * parent : 이 모듈을 import한 모듈 객체
> * children : 이 모듈이 import한 모듈 객체

## import
> * 전역 메소드인 global.require 메소드를 통해서 다른 모듈을 import 한다.
> * unpack을 통해 원하는 요소만 추출해서 사용할 수 있다.
> * ES6부터는 import문을 사용할 수 있다.
> ~~~js
> const { odd, even } = global.require(파일명); // exports가 객체
> const My_function = global.require(파일명); // exports가 함수
> const http = global.require('http'); // exports가 객체
> ~~~
> ~~~js
> import { odd, even } from 파일명;
> import My_function from 파일명;
> import http from 'http';
> ~~~

## export
> * module.exports property에 모듈화 시키고 싶은 객체, 함수, 변수를 넣으면 모듈화가 된다.
> * module을 제외한 exports 단독으로 사용해도 되는데, 이때는 객체의 형식으로만 모듈화가 된다.
> * module.exports와 exports는 동일한 객체를 참조한다.
> ~~~js
> exports.odd = '홀수입니다';
> exports.even = '짝수입니다';
> ~~~
> ~~~js
> module.exports = {
>     odd: '홀수입니다',
>     even: '짝수입니다'
> };
> ~~~
> ~~~js
> obj = {
>     odd: '홀수입니다',
>     even: '짝수입니다'
> };
> export default obj;
> ~~~
