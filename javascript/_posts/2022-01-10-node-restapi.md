---
layout: post
title: [node] ajax and rest api
sitemap: false
---

**참고**
[1] <https://www.ibm.com/cloud/learn/rest-apis>  
* * *  

* toc
{:toc}

## AJAX
> * AJAX(Asynchronous Javascript And XML)는 빠르게 동작하는 동적인 웹 페이지를 만들기 위한 개발 기법. 
> * 웹 페이지 전체를 다시 로딩하지 않고도, 웹 페이지의 일부분만을 갱신할 수 있다.
> * 서버와 통신하여, 원하는 부분만의 데이터를 얻어 갱신할 수 있다.
> * 웹 페이지가 로드된 후에도 자유롭게 서버와 데이터 요청과 응답을 받을 수 있다.
> * 클라이언트가 서버에 데이터를 요청하는 클라이언트 풀링(사용자가 원하는 정보를 서버에게 요청하는 방식)이다. 서버 푸시(사용자가 원하지 않아도 서버가 알아서 자동으로 정보를 제공하는 방식)의 실시간 서비스는 만들 수 없다.
> * 바이너리 데이터를 보내거나 받을 수 없다.
> * jQuery나 axios 같은 라이브러리를 이용해서 AJAX를 하면 된다.
> * axios.get : 서버의 자원을 가져오고자 할 때 사용한다.
> * axios.post : 서버에 자원을 새로 등록하고자 할 때 사용한다.
> * axios.put : 서버의 자원을 치환하고자 할 때 사용한다.
> * axios.patch : 서버의 자원을 일부부만 수정하고자 할 때 사용한다.
> * axios.delete : 사버의 자원을 삭제하고자 할 때 사용한다.

## REST API
> * REST API는 REST(REpresentational State Transfer) 아키텍쳐 스타일의 디자인 원칙을 준수하는 api이다.
> * RESTful API라고 불리기도 한다.
> * 아래의 6가지 조건을 만족해야만 한다.
> 1. **균일한 인터페이스** : 요청이 어디에서 온ㄴ지와 무관하게, 동일한 리소스에 대한 모든 API 요청은 동일하게 보여야 한다. 즉, 서버의 어떤 하나의 리소스에는 하나의 URI(Uniform Resource Identifier)에 속해야한다. (URI는 인터넷에 있는 자원을 나타내는 유일한 주소이다.
> 2. **클라이언트-서버 디커플링** : 클라이언트와 서버간에는 완전히 서로 독립적이어야 한다. 클라이언트가 알아야 하는 유일한 정보는 URI이며, 다른 방법으로는 서버와 상호작용할 수 없다. 서버에서는 HTTP를 통해 요청된 데이터를 전달하는 것 말고는 클라이언트의 프로그램들을 수정할 수 없다.
> 3. **stateless** : 서버는 클라이언트 요청과 관련된 데이터를 저장할 수 없다.
> 4. **캐시 가능성** : 리소스를 캐싱할 수 있어야 한다.
> 5. **계층 구조 아키텍쳐** : 요청과 응답은 다양한 계층을 통과한다. 어플리케이션층 또는 중개자와 통신하는지의 여부를 클라이언트나 서버가 알 수 없도록 해야한다.
> 6. **code on demand** : 일반적으로는 정적 리소스를 전송하지만, 특정한 경우에는 응답에 실행 코드를 포함할 수도 있다. 이러한 경우에 코드는 재 요청시에만 실행시킬 수 있다.