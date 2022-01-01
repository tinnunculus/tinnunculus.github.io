---
layout: post
title: child_process module
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## child_process
> * 노드에서 다른 프로그램을 실행하고 싶을 때, 심지어 다른 언어의 프로그램을 실행시키고 싶을 때 사용할 수 있는 모듈이다.
> * exec은 셸을 실행해서 명령어를 수행하고, spawn은 새로운 프로세스를 띄우면서 명령어를 실행한다.
> ~~~js
> const exec = require('child_process').exec;
> const process = exec('dir');
> process.stdout.on('data', (data) => { console.log(data.toString()); });
> ~~~
> ~~~js
> const spawn = require('child_process').spawn;
> const process = spawn('python', ['test.py']);
> process.stdout.on('data', (data) => { console.log(data.toString()); });
> ~~~