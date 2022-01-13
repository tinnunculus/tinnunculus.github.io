---
layout: post
title: [js] formdata
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## FormData
> * HTML의 form 태그를 동적으로 제어할 수 있다.
> * key, value의 구조로 되어 있다.
> ~~~js
> const formData = new FormData();
> formData.append('name', 'zerocho');
> formData.append('name', 'jongyeon');
> formData.append('birth', 1994);
> formData.has('name'); // true
> formData.get('name'); // zerocho
> formData.getall('name'); // ['zerocho', 'jongyeon']
> ~~~
