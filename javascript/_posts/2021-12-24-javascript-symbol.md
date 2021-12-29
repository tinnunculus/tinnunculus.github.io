---
layout: post
title: symbol
sitemap: false
---

**참고**  
[1] <https://ko.javascript.info/symbol>    
* * *  

* toc
{:toc}

## Symbol
> * 자바스크립틑 객체의 property나 method의 key로는 오직 **문자형과 심볼형**만 가능하다.
> * 심볼은 **유일한 식별자**(unique identifier)를 만들고 싶을 때 사용한다.
> * 심볼을 만들 때는 이름도 붙일 수 있다.
> * 이름이 동일한 심볼을 만들어도 심볼 값은 다르다.
> * 심볼은 문자형으로 자동 형 변환이 일어나지 않는다. 자바스크립트에서 심볼은 **다른 형으로 변환되지 않도록 해야한다**.
> * 출력을 원하면 이름을 통해 출력을 하자.
> ~~~js
> let id1 = Symbol("id");
> let id2 = Symbol("id");
> alert(id1 == id2); // false
> alert(id1); // error.. 심볼은 문자형으로 자동 형 변환이 되지 않는다.
> alert(id1.descripton); // "id"
> ~~~

## hidden property
> * 심볼을 이용하면 hidden property를 만들 수 있다.
> * hidden property는 외부 코드에서 접근할 수도 없고 값도 덮어 쓸 수 없다. // 값을 덮어 쓸 수 없다는 것은 아마 외부코드에서만 말하는 것 같다. 내부에서는 잘 수정한다.
> * 그렇기 때문에 다른 코드로 부터 불러온 객체를 쓸 때, 다른 사람은 모르게 나만의 property를 넣어서 사용할 수 있다.
> * 심볼은 **대괄호를 사용해 키**를 만들어야만 한다. 접근할 때도 대괄호를 통해 접근해야만 한다. obj[symbol]
> * 심볼은 for...in 반복문에서 제외된다. Object.keys(obj)에서도 심볼형 키는 제외된다.
> * Object.assign은 심볼형 키를 제거하지 않고 모든 property를 복제한다.
> ~~~js
> let id = Symbol("id");
> let user = {
>     name: "John",
>     age: 30,
>     [id]: 12345
> };
> for (let key in user) alert(key); // name, age
> let clone = Object.assign({}, user);
> alert( clone[id] );
> ~~~

## 전역 심볼 레지스트리
> * 동일한 이름의 심볼을 생성하더라도 다른 심볼이 생성된다.
> * 심볼을 전역 심볼 레지스트리에 생성하면 **동일한 이름**의 **동일한 심볼**을 생성할 수 있다.
> * 전역 심볼 레지스트리에 있는 심볼을 읽거나, 새로운 심볼을 생성하고 싶다면 Symbol.for(name)를 사용하면 된다.
> * 레즈스트리에 심볼이 존재하면 해당 심볼을 리턴하고, 존재하지 않으면 새로운 심볼을 생성하고 리턴한다.
> ~~~js
> let id = Symbol.for("id"); // 새로운 전역 심볼 생성
> let idAgain = Symbol.for("id"); // 기존의 "id" 심볼
> console.log( id === idAgain );
> ~~~