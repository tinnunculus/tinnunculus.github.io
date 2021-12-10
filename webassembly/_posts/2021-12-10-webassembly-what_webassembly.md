---
layout: post
title: what is Webassembly
sitemap: false
---

**참고**  
[1]. <https://hacks.mozilla.org/2017/02/a-cartoon-intro-to-webassembly/>  
* * *  

* toc
{:toc}
<p align="center"><img src="/assets/img/paper/uncertainty_in_deep_learning/1.png"></p>

기존에는 브라우저를 제어하는 프로그래밍 언어는 자바스크립트 뿐이었다.

*자바스크립트 역사* 
자바스크립트는 1995년에 생겼으며, 속도를 목적으로 디자인되지 않았다.
2008년 부터 많은 브라우저 들이 jit compiler를 추가했고
덕분에 자바스크립트가 실행될 때, JIT는 코드의 패턴을 분석하고 최적화를 통해서 코드를 더 빠르게 실행시킬 수 있었다.