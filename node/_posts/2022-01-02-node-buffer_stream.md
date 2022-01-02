---
layout: post
title: buffer and stream
sitemap: false
---

**참고**  
* * *  

* toc
{:toc}

## buffer
* 파일을 읽거나 쓸 떄처럼, 데이터를 주고 받을떄 데이터를 임시로 저장하는 공간이다. 주로 큐를 통해 만들 것이다.
* 노드에서는 버퍼를 이용하여 파일을 읽거나 쓸 때, 파일의 크기만큼의 버퍼를 생성하고 해당 버퍼에 데이터를 저장한다.
* Buffer 클래스를 이용하면 버퍼를 직접다룰 수 있다.
* 버퍼를 이용해서 데이터를 주고 받는다면 파일 전체가 버퍼에 다 찰때까지 데이터가 전송이 안되므로 스트리밍에 비해 비효율적이다.
~~~js
let buffer = Buffer.from('i like you');
console.log(buffer); // <Buffer 69 20 6c 69 6b 65 20 79 6f 75>
console.log(buffer.toString()); // 'i like you'
buffer = Buffer.alloc(10); // <Buffer 00 00 00 00 00 00 00 00 00 00>
~~~

## streaming
* 버퍼의 크기를 작게 만든 후 여러번 나눠 보내는 방식이다.
* 100MB의 데이터를 1MB 크기의 버퍼로 100번을 나눠 보내는 방식이다.
* 나눠진 조각 데이터를 chunk라고 한다.
* fs 모듈에서 읽고 쓸때 stream을 이용할 수 있으며, 데이터 조각 chunk가 전달될 떄마다 혹은 데이터가 모두 전달될 때 이벤트가 발생하는 이벤트 리스너이다.
~~~js
const fs = require('fs').promises;
/////
const readStream = fs.createReadStream('./readme.txt', { highWaterMark: 16}); // 버퍼의 크기 16
const writeStream = fs.createWriteStream('./readme.txt', { highWaterMark: 16});
const data = [];
/////
readStream.on('data', (chunk) => {
    data.push(chunk);
});
readStream.on('end', () => {
    data = Buffer.concat(data);
});
readStream.on('error', (err) => {
    console.error(err);
});
//////
writeStream.write('이 글을 추가 합니다.');
writeStream.write('이것 또한 추가 합니다.');
writeStream.end(); // finish event call
~~~
* 두개 이상의 스트림을 파이프라인으로 연결해서 연속적으로 데이터를 전달할 수도 있다.
~~~js
const fs = require('fs').promises;
const zlib = require('zlib');
/////
const readStream = fs.createReadStream('readme.txt');
const zlibStream = zlib.createGzip();
const writeStream = fs.createWriteStream('writeme.txt');
readStream.pipe(zlibStream).pipe(writeStream); // 읽고 압축하고 쓰기
~~~