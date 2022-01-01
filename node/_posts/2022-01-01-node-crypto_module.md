---
layout: post
title: crypto module
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## crypto 모듈
> * 문자열을 암호화 시켜주는 모듈이다.
> * 비밀번호같은 중요한 정보는 디비에 저장할 때, 반드시 암호화하여 저장해야만 한다.
> * 이미지도 암호화해서 저장할까?

## 단방향 암호화
> * 복구할 수 없는 암호화를 뜻한다.
> * 비밀번호 같은 경우 복구할 필요가 없기 때문에 단방향 암호화를 사용한다.
> * 단방향 암호화는 주로 해시 기법을 사용한다.
> * createHash(알고리즘) : 사용할 해시 알고리즘을 넣는다. md5, sha1, sha256, sha512 등등이 있다. sha512를 사용하자
> * update(비밀번호) : 암호화할 문자열을 입력한다.
> * digest(인코딩) : 인코딩할 알고리즘을 넣는다. base64, hex, latin1 등이 있고 주로 사용되는 것은 base64이다. 아마 64진수를 나타내는 것이 아닐까 생각하고 64진수로 나타내면 문자열의 길이가 짧아지기에 편리하다.
> ~~~js
> const crypto = require('crypto');
> let password = crypto.createHash('sha512').update('123456789').digest('base64');
> ~~~
> * 현재는 주로 pdkdf2 알고리즘을 이용하여 비밀번호를 암호화한다. 이것은 기존 비밀번호에 salt 라고 불리는 랜덤 문자열을 붙인 후 해시 알고리즘을 반복 적용한다.
> * crypto.randomBytes 이벤트 리스너를 이용하여 주어진 길이의 랜덤 바이트를 생성한다. 이것을 salt라 부른다.
> * 랜덤 바이트 생성과 암호화 처리는 모두 백그라운드에서 별도의 스레드를 이용하여 처리된다.
> * salt를 잘 보관하고 있어야 비밀번호를 찾을 수 있다.
> ~~~js
> const crypto = require('crypto');
> crypto.randomBytes(64, (err, buf) => {
>     const salt = buf.toString('base64');
>     crypto.pbkdf2('123456789', salt, 100000, 64, 'sha512', (err, key) => {
>         console.log("암호화된 비밀번호 : " + key.toString('base64'));
>     }); // 100000번 적용
> });
> ~~~

## 양방향 암호화
> * 암호화된 문자열을 복구하기 위해서는 양방향 암호화를 사용해야만 한다.
> * 암호화할 때 사용했던 키를 가지고 있어야만 복구할 수 있다.
> * 암호화 관련 메소드
> * crypto.createCipheriv(알고리즘, 키, iv) : 암호화할 알고리즘과 키 그리고 암호화시 초기화할 벡터 iv를 입력한다.
> * cipher.update(문자열, 인코딩, 출력 인코딩) : 문자열의 인코딩 정보와 출력할 인코딩 정보를 대입한다.
> * cipher.final(출력 인코딩) : 출력 결과물의 인코딩을 넣으면 암호화가 완료된다.
> * 복구 관련 메소드
> * crypto.createDecipheriv(알고리즘, 키, iv) : 복구할 알고리즘과 키 그리고 복구시 사용할 벡터 iv를 입력한다.
> * decipher.update(암호화된 문자열, 인코딩, 출력 인코딩) : 암호화된 문자열과 그 문자열의 인코딩, 복호화할 인코딩을 넣는다.
> * decipher.final(출력 인코딩) : 복호화 결과물의 인코딩을 넣는다.
> ~~~js
> // 실제 사용할 때는 어떤 알고리즘 사용하는 지, 키가 무엇인지, iv가 무엇인지 이렇게 코드에 다 적어도 되나?? 그냥 npm 라이브러리 알아보는게 나을듯..
> const crypto = require('crypto');
> const algorithm = 'aes-256-cbc';
> const key = 'abcdefgznbkermflwemrl123456';
> const iv = '1234567890123456';
> // 인코딩
> const cipher = crypto.createCipheriv(algorithm, key, iv);
> let encoded = cipher.update('123456789', 'utf8', 'base64') + cipher.final('base64');
> // 디코딩
> const decipher = crypto.createDecipheriv(algorithm, key, iv);
> let decoded = decipher.update(result, 'base64', 'utf8') + decipher.final('utf8');
> ~~~