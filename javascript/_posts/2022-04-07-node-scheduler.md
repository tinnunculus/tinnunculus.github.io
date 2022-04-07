---
layout: post
title: Node- node-schedule
sitemap: false
---

**참고**  
[1] <https://www.npmjs.com/package/node-schedule>  
* * *  

## Jobs
> * 모든 스케줄은 **Job** 객체로 표현된다.
> * Job 객체를 만든 후에 schedule() 메소드를 통해 스케줄을 실행시킬 수 있다.
> * 혹은 scheduleJob()이라는 함수를 사용해서 스케줄을 실행할 수 있다.

## scheduleJob() 함수
> * schedule.scheduleJob(시간, 콜백함수): 파라미터의 시간이 되면 콜백함수가 실행된다.
> * 시간은 총 6개의 term으로 이루어진다.
> * (초) (분) (시) (일) (달) (day of week 0 - 7 (0 or 7 is Sun))
> ~~~js
> const job = schedule.scheduleJob('42 * * * *', function(){
>     console.log('이 콜백 함수가 실행돼요. when the minutes is 42')
> })  
> ~~~
> * 콜백 함수에 파라미터를 받을 수 있다. 실행 예정의 한 시간을 담고 있다.
> ~~~js
> const job = schedule.scheduleJob('0 1 * * *', function(fireDate){
>     console.log('This job was supposed to run at ' + fireDate + ', but actually ran at ' + new Date());
> }
> ~~~
> * Date() 기본 객체를 시간 정보로 입력할 수도 있다.
> ~~~js
> const date = new Date(2012, 11, 21, 5, 30, 0);
> const job = schedule.scheduleJob(date, function(y){
>     console.log(y);
> });
> ~~~

## RecurrenceRule() 함수
> * 반복적으로 스케줄 함수를 실행시키고 싶을 때 사용하는 함수이다.
> ~~~js
> const rule = new schedule.RecurrenceRule();
> rule.minute = 42;
> const job = schedule.scheduleJob(rule, function(){
>     console.log('executes the function every hour at 42 minutes after the hour');
> });
> ~~~
> ~~~js
> const rule = new schedule.RecurrenceRule();
> rule.dayOfWeek = [0, new schedule.Range(4, 6)];
> rule.hour = 17;
> rule.minute = 0;
> const job = schedule.scheduleJob(rule, function(){
>   console.log('message on every Thursday, Friday, Saturday, and Sunday at 5pm');
> });
> ~~~