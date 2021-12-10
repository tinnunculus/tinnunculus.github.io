---
layout: post
title: Javascript in web browser
description: >
  자바스크립트의 기초에 대해서 알아보자.
hide_description: true
sitemap: false
---

* toc
{:toc}

## 웹페이지 렌더링 절차
> <p align="center"><img src="/assets/img/javascript/elementary/1.png"></p>
> * 웹 브라우저는 HTML파일과 CSS파일을 해석(파싱)하여 이미지를 그려낸다.
> * HTML코드 내부에 자바스크립트 코드 혹은 소스파일을 넣어서 사용한다.
> * DOM(Document Object Model) API : 동적으로 페이지의 스타일을 정하는 등 HTML과 CSS를 알맞게 조정하는 역할.
> * 자바스크립트는 DOM API를 사용하여 HTML과 CSS를 수정한다.
> * HTML은 브라우저의 그림을 띄워주는 역할, CSS은 스타일을 입히는 역할, 자바스크립트는 브라우저를 컨트롤및 조절을 하는 역할.
> * 때문에 자바스크립트가 HTML, CSS 코드가 실행되기 전에 실행된다면 문제가 발생한다.
> * 자바스크립트 코드는 자바스크립트 엔진에서 실행된다.
> * 즉, HTML과 CSS 파일을 먼저 파싱하고 DOM 트리를 만든이후에 그림을 그린다. 또한 파싱된 자바스크립트 소스코드는 자바스크립트 엔진으로 넘어가서 처리되고, 자바스크립트에서 DOM API를 사용한다면 다시 DOM 트리를 그려서 그림을 다시 그려주는 것 같다.

## 자바스크립트 코드의 실행
> * HTML 내부에 자바스크립트 코드를 넣기 위해서는 **script** 태크를 이용한다.
> ```js
> // 자바스크립트 코드 직접 입력.
> <script> 
>   // javascript code 
> </script>  
> // 자바스크립트 소스 파일 입력
> <script src = 'test.js'></script>
> ```
> * HTML 내부의 코드들은 위에서부터 순서대로 파싱된다.
> * 만약 자바스크립트에서 DOM API를 사용하여 HTML 요소를 조작할 경우, 자바스크립트가 조작 대상인 HTML 코드보다 먼저 파싱된다면 조작할 요소가 존재하지 않기 때문에 젲대로 동작하지 않을 것이다.
> * 예를들면 자바스크립트 코드가 HTML 문서의 **body**가 해석되기 전인 **head** 부분에 로드되고 실행된다면, 에러를 일으킬 수 있다.
> 
> ### 해결 방법 3가지
> > **1. 이벤트 리스너를 사용하기.**  
> > 브라우저가 완전히 로드된 이후에 이벤트가 발생하고, 콜백함수를 통해 자바스크립트 코드 실행. HTML 문서가 모두 로드된 이후에 코드가 실행되기 때문에 에러를 일으키지 않을 것이다.  
> > <br>
> > **2. 외부 자바스크립트 소스의 경우 async 속성 사용하기.**  
> > ```js
> > <script src='test.js' async></script>
> > ```
> > html 코드를 로딩하는 중에 **script** 태그를 만나면 javascript 코드가 모두 로딩될 때까지 html 로딩은 멈추게 되는데, async 속성을 사용하면 비동기 방식으로 멈추지 않고 html 로딩이 계속된다. 즉, 자바스크립트와 html은 모두 동시에 로드되고 작동될 것이다.  
> > <br>
> > **3. Body 태그의 맨 끝네 넣는 방법**  
> > **body** 태그 바로 앞에 스크립트 태그를 넣음으로써 **head, body**가 모두 로딩된 이후에 스크립트가 로딩되도록 한다.  
> > * 1, 3번 방식은 html DOM이 로드되기 전까지는 script의 로딩과 파싱이 완전히 차단된다. 이는 사이트를 느리게 만드는 중요한 성능 문제를 야기할 수 있다.  
> > * 2번 방식의 asyn를 사용하면 자바스크립트 소스파일의 로딩되는 순서를 보장하지 못한다. 순서를 보장하기 위해서는 defer 속성을 사용하면 된다.  
> > ```js
> > // 1, 2, 3번이 순서대로 실행된다는 보장 X
> > <script async src="1.js"></script>
> > <script async src="2.js"></script>
> > <script async src="3.js"></script>  
> > // 1, 2, 3번이 순서대로 실행된다.
> > <script defer src="1.js"></script>
> > <script defer src="2.js"></script>
> > <script defer src="3.js"></script>  
> > ```
