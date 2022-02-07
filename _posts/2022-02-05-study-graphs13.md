---
layout: post
title:  "[논문리뷰] Diachronic embedding for temporal knowledge graph completion. "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:
---

Goel, R., Kazemi, S. M., Brubaker, M., & Poupart, P. (2020, April). Diachronic embedding for temporal knowledge graph completion. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 3988-3995).

## Abstract
Knowledge graphs (KGs) typically contain **temporal facts indicating relationships among entities at different times**. Due to their incompleteness, several approaches have been proposed to infer new facts for a KG based on the existing ones–a problem known as KG completion. KG embedding approaches have proved effective for KG completion, however, they have been developed mostly for static KGs. Developing temporal KG embedding models is an increasingly important problem. In this paper, we build novel models for temporal KG completion through equipping static models with a **diachronic entity embedding function which provides the characteristics of entities at any point in time**. This is in contrast to the existing temporal KG embedding approaches where only static entity features are provided. The proposed embedding function is **model-agnostic and can be potentially combined with any static model**. We prove that combining it with **SimplE**, a recent model for static KG embedding, results in a fully expressive model for temporal KG completion. Our experiments indicate the superiority of our proposal compared to existing baselines.

## 1. Introduction
#### Knowledge graph 
* KG는 node가 entity를 나타내고 labeled edge가 entity 간의 관계를 나타내는 directed graph
* KG completion이란?
  * 기존에 존재하는 edge (fact) 이외의 새로운 fact를 추론하는 과제
  * 주로 static KGs에 대해서 연구가 되어져왔음 
  * **KG embedding 방법론**이 이 task를 해결하는 데 성능이 좋았음. 
    * entity와 각 relation type을 hidden representation으로 mapping하며, 각 tuple (H,R,T)에 대한 점수를 representation에 score function을 적용하여 계산한다. 
    * entity와 relation을 hidden representation으로 어떻게 mapping 하는 지, score function이 무엇인지에 따라 다양한 방법론이 존재한다 .
 
 #### Previous works
 * 주로 timestamp나 time interval에 KG edge를 연동시킨다.
 * KG embedding 접근법은 주로 temporal aspect를 고려하지 않았고, 이러한 접근 방식에 time을 고려함으로 굉장히 성능이 올라감을 보였다 (Dasgupta, Ray, and Talukdar 2018; Ma, Tresp, and Daxberger 2018; Garcıa-Duran, Dumancic and Niepert 2018)
   * 각 timestamp 별로 hidden representation을 고려하고, entity와 relationship뿐만 아니라 timestamp를 활용할 수 있도록 scorefunction을 extend함으로 시간을 고려함
   * 그러나, 현재 시간 기준의 entity feature나 time에 해당하는 entity features를 하나로 통합하여 각 entity의 representation을 static하게 학습하는 것은 부수적이다. 
  
#### Proposals
* Temporal KG completion (TKGC)
* ex) provide score for (Mary, Liked, God Father, 1995)
  * Mary and God Father features on 1995 not focusing on the current features. 
* Diachronic embedding (DE)
  * 주어진 모든 시간에 대해서 entity feature을 제공
  * entity embedding (*DE*): 특정 시간에 entity와 timestamp를 input으로 하여, entity에 대한 hidden representation을 제공하는 함수
  * DE는 model-agnostic하므로, 어떤 static KG embedding이라도 TKGC로 DE를 leverage하여 확장될 수 있다. 
* SimplE (Kazemi and Poole, 2018b)를 기반으로 실험 


## 2. Background and Notation
#### 2.1. Noation
* Lower-case letters: vectors
* Bold lower-case letters: vectors
* Bold upper-case letters: matrices
* **z**[n]: n-th element of a vector z
* ||z||: norm
* **z**^T: transpose 
* **z1**⊗**z2**: vector z such that **z1**[(n-1)x d2+m] = **z1**[n] x **z2**[m] , flattened vector of the tensor of the two vectors
* <**z1**, ..., **zk**>: sum of the element-wise product of the elements of the k vectors
#### 2.2. Temporal KG completion
* *V*: finite set of entities
* *R*: finite set of relation types
* *T*: finite set of timestamps
* *W* ⊂ *v* x *R* x *v* x *T* represent set of all temporal tuples (v,r,u,t) that are facts and KG  *G* is a subset of *W*. 
* Temporal KG completion is inferring *W* from *G*
#### 2.3. Relation Properties
* symmetric: Relation r이 (v,r,u,t) ∈ *W* ⇐⇒ (u,r,v,t) ∈ *W*
* anti-symmetric: Relation r이 (v,r,u,t) ∈ *W* ⇐⇒ (u,r,v,t) ∈ *W*c
* inverse: (v,r*i*,u,t) ∈ *W* ⇐⇒ (u,r*j*,v,t) ∈ *W*c
* entails: (v,r*i*,u,t) ∈ *W* ⇐⇒ (v,r*j*,u,t) ∈ *W*c

#### 2.4. KG Embedding
* Definition 1. 
  * Entity Embedding (EEMB)
    *  *V* → ψ
    *  모든 entity v를 ψ (non-empty tuples의 class)의 hidden representation으로 mapping하는 function
  * Relation Embedding (REMB)
    * entity (relation)의 hidden representation. 
  * KG embedding model에서는 1) EEMB와 REMB 함수를 정의하고 2) EEMB와 REMB를 input으로 받아서 주어진 tuple에 대해서 score (φ(.))을 계산하는 score function을 정의한다. 이 때 hidden representation의 parameter은 data를 통해서 학습한다. 

## 3. Existing Approaches
* Static KG completion
  * TransE
  * TransE와 embedding 방식은 동일하나 score function이 다른 모델: DisMult, Tucker
  * TransE와 EEMB는 동일하나 REMB와 socre function이 다른 경우: RESCAL, CP
  * SimplE
    * CP에서 entity v에 대해서 inflow와 outflow의 정보 차이가 존재함에 기반하여, relation의 inverse 관계에 주목함
    * REMB(r)=![image](https://user-images.githubusercontent.com/60350933/152690783-56af5db4-c70a-4c3c-824c-506a5e81b357.png) for every r. 
    * φ(v, r, u): average of two CP scores- score for (v,r,u) and score for (u, r^(-1), v).
* Temporal KG completion 
  * TTransE
    * adding one more embedding function mapping timestamp to hidden representation
    * TEMB(t)=(Z*t*) for every *t*
    * φ(v, r, u, t) = ![image](https://user-images.githubusercontent.com/60350933/152690885-68e30b20-3232-4ba6-bd9c-82f9ac1f4f1e.png)
   * HyTE
     * TTransE와 동일한 embedding을 가지나, score function이 다르다. 
     * head, relation, tail embedding을 timestamp의 공간으로 project한 후에 TransE를 projected embedding에 적용한다.
   * ConT 
     * Tuker의 확장버전
   * TA-DistMult
     * DistMult의 확장 버전으로, timestamp의 각 character c가 vector로 mapped된 후에, tuple (v,r,u,t)가 r과 t에서의 characters를 고려하여 생성이 된 후에 Z(r,t)가 temporal relation에 대해 sequence의 각 element의 embedding vector들을 LSTM에 넣음으로 계산을 한다. 이후 score function이 적용된다. 

## 4. Diachronic Embedding
Embedding function에서 time을 input으로 고려하는 **diachronic entity embedding** 제안
#### 4.1.Definition 2
* Diachornic entity embedding (DEEMb):
  * (*V*, *T*) → ψ: 각 pair (v,t)를  ψ의 hidden representation 으로 mapping하는 함수
  * KG의 함수마다 DEEMB 함수를 어떻게 사용할 것인지는 달라질 수 있다. 
* DEEMB function used in the study
  * tuple of vectors that can be generalized
  * Vector in DEEMB(v,t)를 ![image](https://user-images.githubusercontent.com/60350933/152691683-a2fd52d8-da0e-4150-b3bc-b7adb5b4baa7.png)라 할 때, 
    ![image](https://user-images.githubusercontent.com/60350933/152691695-fb791c29-c5a6-47c6-813b-4138ebf82c9b.png)
    * **a**v, **w**v, **b**v는 learning parameter를 보유한 entity-specific vector
      * **w**v를 0으로 설정하여 learnable parameter의 수를 줄이고 temporal signal의 overfitting을 피하여 static features를 explicit하게 modeling함을 통해 Eq(1)의 static feature들이 temporal feature을 통해서도 얻어질 수 있다. 
      * **w**v와 **b**v를 학습함으로, 모델은 entity feature를 시간대별로 고려할 수 있을지를 학습하여, 어떤 시간대이든 정확한 temporal prediction을 할 수 있게 된다. 
      * **a**v는 feature의 중요성을 다루는 함수이다.  
    * σ는 activation fuction
      * *sine*을 activation function으로 주로 사용하는데, 그 이유는 sine function하나가 여러개의 on and off state를 모델할 수 있기 때문이다.  
    * entity는 시간에 따라 변화되는 특성이 있고, 변화되지 않는 특성이 있는데, Eq (1)에서 첫번째 γd element는 temporal 특성을 catch하는 것이고 (1-γ)d element는 static 특징을 파악하고자 한다. 이 때 γ은 temporal feature의 비율을 정하는 hyper-parameter이다. 

#### 4.2. Model-agnosticism
* 기존의 Temporal KG embedding에 대한 proposal은 한 가지 모델에만 적용되는 것이었지만, 해당 연구에서 제안하는 temporal embedding의 경우 EEMB를 DEEMB로 변경하여 TransE, DisMult, SimplE, Tucker 등 다양한 모델의 temporal version을 만들 수 있다. 

#### 4.3. Learning
* KG *G*의 fact는 train, test, valid set으로 나뉘어서 학습됨
* Model parameters는 mini-batch로 stochastic gradient descent를 통해서 학습이 됨
* *B*가 train set의 mini-batch라고 할 때, f=(v,r,u,t)∈*B*에 대해서 (v,r,?,t)와 (?,r,u,t)의 2개의 쿼리를 생성함
  * 첫 번째 쿼리에 대하여 v와 n(negative ratio), *V*에서 랜덤하게 선택된 entity로 구성된 candidate answer set C(f,v)을 생성
  * 두 번째 쿼리에 대하여 첫 번째와 유사한 answer set을 생성함
* Cross entropy loss를 최소화
  ![image](https://user-images.githubusercontent.com/60350933/152715159-9868b815-7a0a-46fd-9158-e0442c1780f6.png)

#### 4.4. Expressivity
* 모델이 expressive하지 않으면 응용 분야에 있어서 underfit할 가능성이 높다. 
* **Definition3** 
  * Model with parameters θ is fully expressive when
    * true tuple과 false tuple이 존재할 때, 이 둘을 완벽하게 분류하는 θ가 존재
* Theorem 1.DE-SimplE는 TKGC에 있어서 fully expressive하다 
  * SimplE에서 entity와 각 관계를 2개의 vector로 mapping한다. 
     ![image](https://user-images.githubusercontent.com/60350933/152715444-89eed602-ff8a-4413-9fa6-48272938858c.png)
  * 증명 위한 case-study
    * γ=d이고, ![image](https://user-images.githubusercontent.com/60350933/152715520-ef6ea9d8-3715-44d1-a06e-dbeda3aabc62.png) 이고 ![image](https://user-images.githubusercontent.com/60350933/152715541-616ce392-4511-4111-b1d3-1cde28c07ec3.png) 일 때, ![image](https://user-images.githubusercontent.com/60350933/152715581-2d51f0ca-ed83-4b0b-8a7e-60d73d10c4d2.png) 의 모든 element는 temporal하고, ![image](https://user-images.githubusercontent.com/60350933/152715610-7349fbb3-7c54-4fdf-8663-a3264275daa1.png)의 모든 element는 non-temporal하다.

#### 4.5. Time Complexity
* Diachronic embedding은 time complexity를 변경시키지 않는다. 
* 해당 모델은 O(d)의 time complexity를 가지며, 이 때 d는 size of the embedding이다. 

#### 4.6. Domain Knowledge
* parameter sharing을 통해서 특정 domain knowledge의 type이 embedding에 포함될 수 있다. 
* SimplE에서 domain knowledge를 포함하는 것이 DE-SimplE에도 ported될 수 있다. 
  *  ri ∈ R with ![image](https://user-images.githubusercontent.com/60350933/152716122-43637bf6-59de-4128-a1cb-cdd8b41c33db.png)이라 할 때, r*i*가  symmetric하거나 anti-symmetric이면 embedding을 ![image](https://user-images.githubusercontent.com/60350933/152716164-1d9650a6-eead-481a-bcd1-58b38500dbb1.png) 바꾸거나,![image](https://user-images.githubusercontent.com/60350933/152716189-7210c3ad-328f-47ef-868f-7f20ed102eea.png)에 음수화함으로 knowledge 흡수 가능
  *  r*i*가 inverse이면 ![image](https://user-images.githubusercontent.com/60350933/152716164-1d9650a6-eead-481a-bcd1-58b38500dbb1.png)가 가능함
*  Proposition1. Symmetry, anti-symmetry, inversion은 SimplE와 동일하게 DE-SimplE로 합쳐질 수 있다.
*  Proposition2. ![image](https://user-images.githubusercontent.com/60350933/152726327-8a831ef6-e116-4c43-bfe8-e1f7e03a1e7f.png)를 non-negative로, activation function을 non-negative range (ex. ReLU, sigmoid 등) 로 두면, entailment가 DE-SimplE로 SimplE처럼 incorporated될 수 있다. 

## 5. Experiments & Results
* Datasets
  * ICEWS: two subets generated by the facts in 2014 and facts between 2005 to 2015
  * GDELT: subset corresponding to the facts from April 1, 2015 to March 31, 2016.  
* Baselines
  * static: TransE, DistMult, SimplE where the timing information are ignored
  * temporal in previous sections
* Metrics
  * Mean reciprocal rank (MRR)
    ![image](https://user-images.githubusercontent.com/60350933/152731117-9bdd4b72-9b29-4722-812c-feaa9c866054.png) ![image](https://user-images.githubusercontent.com/60350933/152731288-2223464b-bbce-42c5-b9f1-439ef999395c.png)
    * Mean rank 보다 더 안정적인 방법
  * Hit@k
    ![image](https://user-images.githubusercontent.com/60350933/152731353-196c0044-dcd6-40cf-ac68-49e7e7708e2b.png)
* Implementation
  * Time stamp in the dataset are dates rather than single numbers, temporal part of Eq1 to year, month, day에 각각 적용함. (Three temporal vectors) 
    * 3개의 temporal vector에 대해서 element-wise sum을 취해 single temporal vector을 얻음
    * converting a date into a timestamp in the embedded space. 

#### 5.2. Comparative study
* Temporal versions of different models outperform the static counterparts
![image](https://user-images.githubusercontent.com/60350933/152733686-8ec570a1-3bc9-4349-81ab-0ffc849bc6f3.png)

#### 5.3. Model variants & Ablation study
![image](https://user-images.githubusercontent.com/60350933/152733725-4ff6bbe3-e5b1-4fa2-a4ca-b5348f102432.png)

 * Activation function
   * *sine*을 activation 함수로만 사용하는 것이 아닌 Tanh, sigmoid, Leaky ReLU, squared exponential을 사용하여 performance 비교를 하였음. 
     * 성능도 좋으며, 특히 squared exponential의 성능이 sine과 유사하게 나옴. 
     * 그 이유는 sine이나 square exponential이 더 정교한 feature산출이 가능한데, 탄젠트와 시그모이드는 smooth on-off temporal switch에 대응하는 것이라면 sine이나 squared exponential activation은 2개 이상의 switch (ex. on-off-on)와 유사한 역할을 하여, 특정 시간에 시작하고 일정 시간 후에 종료되는 관계에 대한 설명이 잘 되는 것이기 때문이다. 
 
* Diachronic Embedding for Relation 
  * Entity에 비해 relation이 천천히 evolve한다는 가설을 세웠음
  * DE-TransE, DE-DistMult에서 relation embedding이 time의 function이 되도록 설계하였고, entity와 relation 모두가 DE인 경우에 entity에만 DE를 적용한 경우와 성능이 동일함을 알 수 있다. 
  * 따라서 evolution of relation을 modeling하는 것은 의미가 없다는 것을 보여준다. 

* Generalizing unseen timestamps
  * 5, 15, 25일에 해당하는 fact를 제외한 dataset을 생헝하였고, 제외된 fact를 valid와 test data에 추가하였다. 이 데이터들은 train에서는 확인할 수 없다. 
  * DistMult와 DE-DistMult에 적용을 했을 떄 DE-DistMult의 MRR 성능이 원 모델보다 더 성능이 좋음을 알 수 있다. 

* Importance of model parameters
  * Temporal part of the embedding은 a, w, b로 나뉘어진다. 
  * 각각의 중요성에 대해서 파악을 하기 위해서 a,w,b를 제외한 각각의 모델을 테스트 해 봄. 
  * 결과
    * a,w가 특히 중요하고, b를 제외했을 때 큰 영향은 없었다.    

* Static features
  ![image](https://user-images.githubusercontent.com/60350933/152737642-2d9c99a0-f109-4ee4-83b8-1b1754864c8e.png)
  * Feature이 temporal이 되는 경우 (γ가 0에서 non-zero number로 변경)에 성능이 더 좋아짐. 
  * γ (percentage of temporal features) 이 커질 수록 MRR이 더 커지고 줄어든다 (마지막에 줄어드는 것은 overfitting문제일 것) 
  * Static feature을 모델링 하는 것은 learnable parmeter을 줄이고 overfitting을 피할 수 있게 함.  
    * 이는 embedding dimension이 커질수록 더 중요할 것이다.   

* Training curve
  * sine activation function을 사용했을 때 안정적인 학습이 가능함.  

## 6. Conclusion
* Future Studies
  * designing functions other than the one proposed in Eq1
  * A comoprehensive study of which functions are favored by different types of KGs
  * Using proposed embedding for diachronic word embedding.  
