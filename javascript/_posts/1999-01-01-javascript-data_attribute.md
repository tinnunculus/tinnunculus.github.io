---
layout: post
title: Js- data attribute
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## data attribute
* HTML에 자바스크립트에서 보내준 데이터를 저장하는 방법
* HTML 태그의 속성으로 data- 시작하는 것들을 넣는다.
* 자바스크립트에서 쉽게 접근할 수 있고, HTML에서 - 문자는 자바스크립트에서 대문자로 바뀌어 들어간다.
* 자바스크립트 dataset에 데이터를 넣어도 html에 자동 반영이 된다.
~~~html
<ul>
    <li data-id="1" data-user-job="programmer">Zero</li>
    <li data-id="2" data-user-job="designer">Zero</li>
    <li data-id="3" data-user-job="programmer">Zero</li>
    <li data-id="4" data-user-job="ceo">Zero</li>
    <li data-id="5" data-user-job="cso">Zero</li>
</ul>
<script>
    console.log(document.querySeletor('li').dataset); // {id: '1', userJob: 'programmer'}
</script>
~~~