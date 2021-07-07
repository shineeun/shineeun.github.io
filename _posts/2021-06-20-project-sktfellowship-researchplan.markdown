---
layout: post
title:  "[SKT Fellowship] 바른말 고운말 팀 연구계획서"
subtitle:   "바른말 고운말 팀 연구계획서"
categories: project
tags: sktfellowship
comments: true
header-img:
---



안녕하세요.

저희는 **KoBERT/KoGPT/KoBART** **기반 언어처리** **Application** **개발**이라는 연구 과제에 참여하게 된 SKT AI Fellowship 3기 바른말 고운말 팀입니다.

이번 글을 통해 저희의 **연구 과제 계획**을 전달 드리려고 합니다.





## 1. 연구과제 배경 및 목표 

![image](https://user-images.githubusercontent.com/47618340/122662345-399ca480-d1cd-11eb-8c0e-567135b5cff6.png)

---



## 2. 만들고자 하는 Output

1. 오픈 API 공개
   + API를 공개하여 커뮤니티 등의 서비스를 제공하는 기업에서 사용할 수 있도록 하는 것입니다.



2. 웹 데모 제작
   + URL 입력 시 해당 URL에 있는 글에서 혐오표현을 탐지하고, 해당 표현을 순화된 표현으로 변환시켜 보여주는 데모 제작하는 것입니다.

<img src="https://user-images.githubusercontent.com/47618340/122662366-5f29ae00-d1cd-11eb-9b08-b6680c31f56c.png" alt="image" style="zoom: 200%;" />

---



## 3. 사용 데이터

저희가 **사용하고자 하는 데이터**는 Github Korean Hate Speech, data를 활용할 예정입니다.

혐오표현이라고 탐지 하지 않은 데이터 중에서 hand-labeling을 추가로 진행하고자 합니다.



![image](https://user-images.githubusercontent.com/47618340/122662373-6a7cd980-d1cd-11eb-831e-0c060cc0a886.png)



---

## 4. 모델링 (1)  - 혐오표현 탐지

1. Baseline 모델 구축: Ko-BERT

2. Knowledge Graph 활용하여 문맥 고려한 Ko-BERT 응용 모델 구축

![image](https://user-images.githubusercontent.com/47618340/122662377-7072ba80-d1cd-11eb-99e3-baeb229cf057.png)



---

## 5. 모델링 (2) - Text Style Transfer

2개의 모델을 구축하여 성능이 좋은 모델 활용 예정입니다.



### (1) TST with Parallel supervised data

![image](https://user-images.githubusercontent.com/47618340/122662381-75d00500-d1cd-11eb-9fea-496a64f31bf0.png)



> **Similarity 계산 방법**
>
> + BERT-embedding + Sen2Vec, Word2Vec
> + ANNOY (Approximate Nearest Neighbors Oh Yeah)



**모델**

+ Seq2seq 모델인 Ko-BART 활용

  ![image](https://user-images.githubusercontent.com/47618340/122662385-7bc5e600-d1cd-11eb-9caa-20927d6d0d42.png)





### (2) TST with Non-Parallel supervised data

+ Non-parallel data: data without any knowledge of matching text pairs in different styles



![image](https://user-images.githubusercontent.com/47618340/122662387-82545d80-d1cd-11eb-9ef7-34d03d9e5f47.png)



---

## 6. 연구 과제 가치

저희가 고안한 연구 과제의 가치는 다음과 같습니다.



![image](https://user-images.githubusercontent.com/47618340/122662392-87b1a800-d1cd-11eb-8e4e-818b6dfc6e83.png)

앞으로의 5개월 동안 계획한 연구를 잘 이루어 나갈 수 있도록 열심히 공부하고, 코드 짜겠습니다 재밌고 즐겁게 SKT AI Fellowship 만들어 나갈게요. 지금까지 많은 것을 알려주신 8팀 권득신 멘토님 정말 감사드리고, 박하은, 이민정 담당자님 여러 모로 많은 도움 주셔서 감사합니다. 앞으로 잘 부탁드립니다!



궁금하신 사항이나 피드백 주실 사항 있으시다면 lhmlhm1111@yonsei.ac.kr로 연락 부탁드립니다!

