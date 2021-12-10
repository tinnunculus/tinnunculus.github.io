---
layout: post
title: Promise
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/async>  
* * *  

* toc
{:toc}

## Promise란..?
> * Promise는 기본적으로 객체이다.
> * Promise는 함수(excutor)를 인자로 받고 객체가 생성될 때, 해당 함수가 실행된다.
> * excutor 함수는 두개의 **함수**를 인자로 받는다.
> * resolve와 reject는 자바스크립트에서 제공하는 함수이다.
> * Promise는 성공 또는 실패만 한다. 따라서 executor 함수 내에서 resolve(성공)와 reject(실패) 함수 둘 중 하나는 반드시 콜 해야한다.
> * 또한 한번만 호출되어야 한다. 두번 이상 호출되면 두번째 호출부터는 실행되지 않는다.
> * resolve 함수와 reject 함수는 기본적으로 callback 함수이다.
> * excutor 함수가 실행되고 내부에서 resolve, reject 함수가 실행되면 resolve 함수와 reject함수는 background에 보내지고 바로 스택에 쌓인다.
> ~~~js
> // resolve와 reject는 callback 함수이기 때문에 then 메소드는 바로 실행되지 않고 background로 넘어간 이후에 비동기로 실행된다. 따라서 console에 2, 5, 4, 1 순서로 찍힌다.
> test = new Promise((resolve, reject) => {
>     console.log(2);
>     resolve(1);
> });
> console.log(5);
> test.then((result) => {console.log(result);});
> console.log(4);
> ~~~
> * resolve(value) - 일이 성공적으로 끝난 경우 그 결과를 나타내는 value와 함께 호출.
> * reject(error) - 에러 발생 시 에러 객체를 나타내는 error와 함께 호출.
> ~~~js
> // resolve 호출
> let promise = new Promise(function(resolve, reject){
>     setTimeout(() => resolve("done"), 1000);    
> });
> // reject 호출
> let promise2 = new Promise(function(resolve, reject){
>    setTimeout(() => reject(new Error("error")), 1000);
> })
> ~~~
> 
> ### Promise 객체의 주요 property
> > <p align="center"><img width="550" src="/assets/img/javascript/promise/1.png"></p>
> > * **state** : 문자열 데이터이며, 처음에는 "pending" 이었다가 excutor에서 resolve 함수가 호출되면 "fulfilled", reject 함수가 호출되면 "reject"로 변환된다.
> > * **result** : 처음에는 undefined 이며, resolve가 호출되면 value, reject가 호출되면 error를 반환한다.
> > * 하지만 state, result property 모두 직접 접근할 수 없다. then, catch, finally 메소드를 사용하여 상태나 결과값에 접근해야 한다.

## Promise 객체의 then, catch, finally 메소드
> * Promise 객체는 excutor 함수의 실행 결과를 처리하기 위해, then, catch, finally 3개의 메소드를 가진다.
> 
> ### then 메소드
> > * 두개의 함수를 인자로 받는다. 각각의 함수는 인자를 하나씩 가지고 있고, 첫번째 함수는 promise가 resolve 되었을 때 실행되는 함수이고, 두번째 함수는 promise가 reject 되었을 때 실행되는 함수이다.
> > ~~~js
> > let promise = new Promise(function(resolve, reject){
> >     setTimeout(() => resolve("done!"), 1000);
> > });
> > promise.then(result => alert(result), error => alert(error));
> > /// 혹은
> > promise.then(result => alert(result)); // 작업이 성공된 경우에만 다루고 싶다면!! 하나만 적어도 유효
> > /// 두개의 .then을 사용한다면 두개다 실행된다.
> > ~~~
> > ~~~js
> > let promise = new Promise(function(resolve, reject){
> >     setTimeout(() => reject(new Error("error")), 1000);
> > });
> > promise.then(result => alert(result), error => alert(error));
> > promise.then(error => alert(error)) // 오류!! 작업이 실패한 경우를 then으로 다루고 싶아면 항상 두 인자 모두 적어줘야함.
> > ~~~
> > * resolve(1) 함수가 일찍 콜 되었다고 해서 then 함수가 resolve(1) 뒤에 있는 코드들보다 빨리 실행되지는 않는다.
> > * 주의점) 콜백 함수로 에러 처리를 했을때(error first callback), 에러 인자를 왼쪽에 썼었는데 then 함수는 오른쪽에 에러 인자로 사용한다.
> > ~~~js
> > let promise = new Promise((resolve, reject) => {
> >     resolve(1);
> >     // ...
> > }).then(result => {...})
> > ~~~
> 
> ### catch 메소드
> > * 에러가 발생한 경우만 다루고 싶다면, **then(null, new Error)** 같이 첫번째 인자를 **null**로 전달하면 되는데, 이와 같은 구문을 catch(new Error)를 써도 같은 작동을 합니다. 
> > ~~~js
> > let promise = new Promise(function(resolve, reject){
> >     setTimeout(() => reject(new Error("error")), 1000);
> > })
> > promise.catch(error => alert(error));
> > ~~~
> 
> ### finally 메소드
> > * promise의 성공과 실패의 상관없이 마지막에 실행이 됨. 인자는 함수 하나를 받는데 해당 함수는 인자를 갖지 않음(전달 받는게 없다).
> > ~~~js
> > let promise = new Promise(function(resolve, reject){
> >     setTimeout(() => reject(new Error("error")), 1000);
> > });
> > promise.then(result => alert(result));
> > promise.catch(error => alert(error));
> > promise.finally(() => alert("promise 이행 완료"));
> > ~~~

## callback -> promise 예시
> * 기존의 콜백 함수를 통해 체인 구조를 이루면, 코드가 복잡해지는 문제가 있는데, 이를 promise를 통해 어느정도 해결할 수 있다.
> ~~~js
> // callback 
> function loadScript(src, callback){
>     let script = document.createElement('script');
>     script.src = src;
>     script.onload = () => callback(null, script);
>     script.onerror = () => callback(new Error("에러발생"));
>     document.head.append(script);
> }
> loadScrript('test.js', (error, script) => {alert(script.src);});
> // promise
> // resolve 함수와 reject 함수 두개가 실행된 것처럼 보이지만 onload와 onerror는 이벤트 기반의 함수이기 때문에 둘 중 하나만 call 된다. 
> function loadScript(src){
>     return new Promise(function(resolve, reject){
>         let script = document.createElement('script');
>         script.src = src;
>         script.onload = () => resolve(script);
>         script.onerror = () => reject(new Error("error")); 
>         document.head.append(script);
>     })
> }
> let promise = loadScript("test.js");
> promise.then(script => alert("good"));
> promise.catch(error => alert(error));
> ~~~

## 프라미스 체이닝
> * 비동기 작업을 순차적으로 처리하도록 함.
> * 원래는 excutor 함수 내에서 resolve 함수나 reject 함수가 호출되면 then 메소드의 인자로 들어간 함수가 호출된다고 했었다.
> * 아래 코드에서 보면 then 메소드의 인자 함수는 value를 리턴한다. 하지만 then 함수는 promise 객체를 리턴한다.
> * 또한 then 메소드의 인자 함수가 value를 리턴한다면 then함수는 리턴한 promise 객체의 resolve(value) 함수를 call해서 다음 then 함수가 호출되도록 할 수 있다.
> * 따라서 아래 코드는 console에 2, 4, 8이 찍힌다.  
> ~~~js
> new Promise(function(resolve, reject) {
>     setTimeout(() => resolve(1), 1000);
> }).then(function(result){
>     console.log(result);
>     return result * 2;
> }).then(function(result){
>     console.log(result);
>     return result * 2;
> }).then(function(result){
>     console.log(result);
>     return result * 2;
> })
> ~~~
> * 첫번째 예시에서는 then 메소드의 인자 함수는 값(value)이 리턴됐다. 하지만 그 다음 then 메소드도 호출되었다. 이것은 내부적으로 then 메소드는 Promise 객체를 생성해서 resolve(value) 함수를 실행하기에 가능하다. 아래 예시는 그것을 실제 구현한 코드이다.
> ~~~js
> new Promise(function(resolve, reject) {
>     setTimeout(() => resolve(1), 1000);
> }).then(function(result){
>     console.log(result);
>     return new Promise((resolve, reject) => {
>         resolve(result*2);
>     })
> }).then(function(result){
>     console.log(result);
>     return new Promise((resolve, reject) => {
>         resolve(result*2);
>     })
> }).then(function(result){
>     console.log(result);
>     return new Promise((resolve, reject) => {
>         resolve(result*2);
>     })
> })
> // 첫번째 예시와 동일한 프로세스.
> ~~~
> <p align="center"><img width="550" src="/assets/img/javascript/promise/2.png"></p>
> * 아래 코드는 체인으로 연결되지 않는다. 독립적으로 처리할 뿐이다.
> ~~~js
> let promise = new Promise(function(resolve, reject){
>     setTimeout(() => resolve(1), 1000);
> });
> promise.then(function(result){
>     alert(result); // 1
>     return result * 2;
> });
> promise.then(function(result){
>     alert(result); // 1
>     return result * 2;
> });
> promise.then(function(result){
>     alert(result); // 1
>     return result * 2;
> });
> ~~~
> 
> ### Promise 체이닝 추가 예시
> > ~~~js
> > * 아래 코드는 Promise 객체를 변수에 저장하지 않고 바로 생성해서 then 메소드를 붙여준 것.
> > * Promise 객체는 생성과 동시에 excutor 함수가 실행되기 때문에 then 메소드가 순차적으로 콜된다.  
> > new Promise(function(resolve, reject){
> >     setTimeout(() => resolve(1), 1000);  
> > }).then(function(result){
> >     alert(result); // 1
> >     return new Promise((resolve, reject) => {
> >         setTimeout(() => resolve(result*2), 1000);  
> >     });
> > }).then(function(result){
> >     alert(result); // 2
> >     return new Promise((resolve, reject) => {
> >         setTimeout(() => resolve(result*2), 1000);
> >     });  
> > }).then(function(result){
> >     alert(result); // 4
> >     return new promise((resolve, reject) => {
> >         setTimeout(() => resolve(result*2), 1000);
> >     });  
> > });
> > ~~~
> > * 아래 코드는 test1.js, test2.js, test3.js를 비동기 순차적으로 로딩을 한다.
> > ~~~js
> > loadScript("test1.js")
> > .then(function(script){
> >     return loadScript("test2.js");  
> > }).then(function(script){
> >     return loadScript("test3.js");  
> > });
> > ~~~

## Thenable
> * Promise의 then 메소드는 Promise 객체를 반환하거나, Value를 반환할 경우 자체적으로 Promise 객체를 생성 및 반환한다.
> * 하지만 Promise객체의 then 메소드가 아닌 then 메소드를 가지는 다른 객체를 반환할 수도 있다.
> * 이러한 객체를 thenable 객체라고 부르며, then 메소드를 가지고 있어야 하며, (resolve, reject) 인자를 가지는 함수여야만 한다.
> * thenable 객체의 then 메소드는 resolve(result)를 호출하거나, reject(error)를 호출해야만 한다.
> ~~~js
> function Thenable(){
>     this.num = 2;
>     this.then = function(resolve, reject){
>         setTimeout(() => resolve(this.num * 2));
>     };
> }
> new Promise(resolve => resolve(1))
> .then(result => {
>     return new Thenable();
> }).then(alert); // Thenable.then(alert) 가 실행되는게 아닌 먼저 thenable.then(resolve, reject)가 실행되고 난 결과물(resolve, reject)이 반환되어 then(result => alert(result)) 함수가 실행된다.
> ~~~

## fetch와 체이닝 함께 응용하기
> * fetch 함수는 url을 입력으로 받고, 해당 url로 네트워크 요청을 보낸다.
> * 해당 url의 서버가 응답을 보내면 fetch 함수는 Promise 객체를 리턴하며, excutor 함수에서 resolve(response) 함수를 실행한다.
> ~~~js
> // "jongyeon/test.js" 주소에서 파일을 읽는 요청을 보내고
> // 해당 파일을 text 형식으로 알림한다.
> fetch('jongyeon/test.js')
> .then(function(response){
>     return response.text(); // response.text() 함수는 Promise 객체를 return 하면서 resolve(text) 함수를 call 한다.
> }).then(function(text){
>     alert(text);
> });
> ~~~  
> * 비동기(Asynchronous)은 항상 promise 객체를 리턴하도록 하는게 좋다.
> * 현재는 chain을 확장할 계획이 없더라도 나중에는 언제라도 chain 확장을 손쉽게 할 수 있기 때문이다.
> * 또한 재사용 가능하도록 함수 단위로 분리하는게 좋다.
> ~~~js
> // "/article/promise-chaining/user.json" 에서 파일을 읽는 요청을 보내고
> // 파일에서 user name의 정보를 읽어 해당 유저의 깃허브에 요청을 보낸다.
> // 깃허브에서 이미지를 읽어와서 창에 보여주자.
> function loadJson(url){
>     return fetch(url).then(response => response.json());
> }
> function loadGithubUser(name){
>     return fetch(`https://api.github.com/users/${name}`).then(response => response.json());
> }
> function showAvatar(githubUser){
>     return new Promise(function(resolve, reject){
>         let img = document.createElement('img');
>         img.src = githubUser.avatar_url;
>         img.className = "promise-avatar-example";
>         document.body.append(img);
>         setTimeout(() => {
>             img.remove();
>             resolve(githubUser);
>         }, 3000);
>     });
> }
> //
> loadScript('/article/promise-chaining/user.json')
> .then(user => loadGithubUser(user.name))
> .then(showAvatar)
> .then(githubUser => alert(`Finished showing ${githubUser.name}`));
> ~~~

## Promise와 에러 처리
> * 콜백 함수를 사용해서 에러처리를 했던 경우에는 error first callback 기법을 사용해서 처리해주었다. 
> * Promise를 사용할 경우 rejection 함수를 사용하여 catch 메소드에서 다루어주면 된다.
> * catch 메소드는 첫번째 핸들러일 필요는 없고 여러 개의 then 뒤에 올 수 있다.
> ~~~js
> fetch('/article/promise-chaining/user.json')
> .then(response => response.json())
> .then(user => fetch(`https://api.github.com/users/${user.name}`))
> .then(githubUser => new Promise(resolve, reject) => {
>   let img = document.createElement('img');
>   img.src = githubUser.avatar_url;
>   img.className = "promise-avatar-example";
>   document.body.append(img);
>   setTimeout(() => {
>     img.remove();
>     resolve(githubUser);
>   }, 3000);
> })
> .catch(error => alert(error.message));
> ~~~
> 
> ### 암시적 try...catch
> > * reject 함수는 암시적으로 try 구문을 쓰는 것과 동일하다.
> > ~~~js
> > new Promise((resolve, reject) => {
> >   reject(new Error("에러 발생"));  // throw new Error("에러 발생"); 과 동일한 코드이다.
> > })
> > ~~~
> > 
> 
> ### 에러 다시 던지기
> > * 일반 try...catch 구문에서는 catch 구문에서 에러를 분석하고, 처리할 수 없는 에러라고 판단이 되면 에러를 다시 던질 때가 있다.
> > * Promise에서도 이와 유사한 일을 할 수 있다.
> > ~~~js
> > new Promise((resolve, reject) =>{
> >   throw new Error("에러 발생");
> > }).catch(function(error) {
> >   if (error instanceof URIError){
> >     // 에러 처리
> >   } else {
> >     alert("처리할 수 없는 에러");
> >     throw error; // 에러 다시 던지기!
> >   }
> > }).then(function() {
> >   // 에러가 잘 처리되었으면 여기로 제어가 이동
> > }).catch(error => {
> >   // 에러가 잘 처리되지 않았으면 여기로 제어가 이동
> > });
> > ~~~

## Promise의 API
> * Promise 객체는 5가지 메소드(all, allSettled, race, resolve, reject)를 가지고 있다.
> 
> ### Promise.all()
> > * 여러 개의 Promise들을 동시에 실행시키고 모든 Promise가 준비될 때까지 기다릴려고 할 때 사용한다.
> > * 예를 들면 여러 개의 URL에 동시에 요청을 보내고, 다운로드가 모두 완료된 후에 일괄적으로 처리할 때 쓸 수 있다.
> > * Promise.all 메소드에 들어가는 순서에 따라 출력이 된다.
> > ~~~js
> > // 3초 후에 resolve([1,2,3])이 실행된다.
> > Promise.all([
> >   new Promise(resolve => setTimeout(() => resolve(1), 1000)),
> >   new Promise(resolve => setTimeout(() => resolve(2), 2000)),
> >   new Promise(resolve => setTimeout(() => resolve(3), 3000))
> > ]).then(alert);
> > ~~~
> > ~~~js
> > // 3개의 url에 동시에 요청을 보내고 응답을 받기.
> > let urls = [
> >   'https://api.github.com/users/iliakan',
> >   'https://api.github.com/users/remy',
> >   'https://api.github.com/users/jeresig'
> > ]
> > let requests = urls.map(url => fetch(url)); // [promise1, promise2, promise3] 이 생성되고 각각의 excutor 함수가 실행된다.
> > Promise.all(requests) // 꼭 Promise 객체가 생성되는 것을 넣지 않아도 된다. 기존 Promise 객체를 넣어도 된다.
> > .then(responses => {
> >   for(let response of responses){
> >     alert(`${response.url}: ${response.status}`); // 정상적으로 응답이 오면 모든 url의 응답코드가 200입니다.
> >   }
> >   return responses;
> > }.then(responses => Promise.all(responses.map(r => r.json())))
> > .then(users => users.forEach(user => alert(user.name)));
> > ~~~
> > * Promise.all에 전달되는 Promise 중 하나라도 거부되면, Promise.all이 반환하는 Promise는 에러와 함께 거부된다.
> > * 에러가 발생하면 정상적으로 이행된 Promise도 무시된다.
> > ~~~js
> > Promise.all([
> >   new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
> >   new Promise((resolve, reject) => setTimeout(() => reject(new Error("에러 발생")))),
> >   new Promise((resolve, reject) => setTimeout(() => resolve(2), 1000))
> > ]).catch(alert);
> > ~~~
> > * Promise.all은 Promise 객체가 아닌 일반 value도 입력으로 받는다. 입력받은 value 들은 그대로 resolve(value)가 실행된다.
> > ~~~js
> > Promise.all([
> >   new Promise(resolve => setTimeout(() => resolve(1), 1000),
> >   2,
> >   3
> > ]).then(alert); // 1, 2, 3
> > ~~~
> 
> ### Promise.allSettled
> > * Promise.all 메소드는 하나의 Promise 라도 이행에 실패하면 전체 Promise가 이행되지 않는다.
> > * 이를 해결하기 위해 나온 것이 **allSettled** 메소드이며, 몇개의 Promise가 이행되지 않더라도 then 메소드가 실행된다.
> > * allSettled 메소드는 결과로 출력된 배열의 요소들은 두개의 property를 가지고 있다.
> > * 응답에 성공했을 경우 - {status:"fulfilled", value:result}
> > * 응답에 실패했을 경우 - {status:"rejected", reason:error}
> > ~~~js
> > let urls = [
> >     'https://api.github.com/users/iliakan',
> >     'https://api.github.com/users/remy',
> >     'https://no-such-url'
> > ];
> > Promise.allSettled(urls.map((url => fetch(url))))
> > .then(results => {
> >     results.forEach((result, num) => {
> >         if (result.status == "fulfilled"){
> >             alert(`${urls[num]}: ${result.value.status}`);
> >         }
> >         if (result.status == "rejected"){
> >             alert(`${urls[num]}: ${result.reason}`);
> >         }
> >     });
> > });
> > ~~~
> 
> ### Promise.race
> > * 여러 Promise를 입력으로 받은 후에 가장 먼저 실행되는(background에서 가장 먼저 위로 콜되는) Promise를 반환한다.
> > ~~~js
> > Promise.race([
> >     new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
> >     new Promise((resolve, reject) => reject(new Error("에러 발생"))),
> >     new Promise((resolve, reject) => resolve(() => resolve(3), 3000))
> > ]).then(alert); // 1
> > ~~~

## Promisification
> * 콜백 기반의 함수를 Promise 어법상에서도 돌아가도록 만들어주는 것.
> ~~~js
> // 콜백 기반의 함수
> // usage :
> // loadScript('test.js', (err, script) => {스크립트가 로드되고 할 일});
> function loadScript(src, callback) {
>     let script = document.createElement('script');
>     script.src = src;
>     script.onload = () => callback(null, script);
>     script.onerror = () => callback(new Error("에러 발생"));
>     document.head.append(script);
> }
> // promisification
> // usage :
> // loadScriptPromise('test.js').then((result) => {스크립트가 로드되고 할 일})
> let loadscriptPromise = function(src){ // src라는 인자를 받아야하기 때문에 Promise 객체가 아닌 function으로 설정
>     return new Promise((resolve, reject) => {
>         loadScript(src, (err, script) => {
>             if (err) {
>                 reject(err);
>             } else {
>                 resolve(script);
>             }
>         });
>     });
> }
> ~~~
> * 아래의 코드는 콜백 함수를 입력 받고 promisification 해주는 함수를 구현한 것이다.    
> ~~~js
> function promisify(f) {
>     return function(...args) {
>         return new Promise((resolve, reject) => {
>             console.log(this);
>             function callback(err, result) {
>                 if (err) {
>                     reject(err);
>                 } else {
>                     resolve(result);
>                 }
>             }
>             args.push(callback);
>             f.call(this, ...args); // 여기의 this는 global이다. 근데 왜 this를 써줬을까...?
>         })
>     }
> }
> let pf = promisify(loadScript);
> pf("test.js").catch(err => {});
> ~~~
> * Promisification는 콜백을 완전히 대체하지는 못한다.
> * Promise는 하나의 결과만 가질 수 있지만, 콜백은 여러 번 호출할 수 있기 때문이다...?
> * 따라서 Promisification은 콜백을 단 한번 호출하는 함수에만 적용해야 한다.