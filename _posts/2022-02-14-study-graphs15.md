---
layout: post
title:  "[논문리뷰] OAG: Toward Linking Large-scale Heterogeneous Entity Graphs "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:
---

Zhang, F., Liu, X., Tang, J., Dong, Y., Yao, P., Zhang, J., ... & Wang, K. (2019, July). Oag: Toward linking large-scale heterogeneous entity graphs. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2585-2595).

Linking entities from different sources is a fundamental task in building open knowledge graphs. Despite much research conducted in related fields, the challenges of linking large-scale heterogeneous entity graphs are far from resolved. Employing two billion-scale academic entity graphs (Microsoft Academic Graph and AMiner) as sources for our study, we propose a unified framework --- **LinKG**--- to **address the problem of building a large-scale linked entity graph**. LinKG is coupled with three linking modules, each of which addresses one category of entities. To link *word-sequence-based entities* (e.g., venues), we present **a long short-term memory network-based method for capturing the dependencies**. To link *large-scale entities* (e.g., papers), we leverage **locality-sensitive hashing and convolutional neural networks for scalable and precise linking**. To link entities with *ambiguity* (e.g., authors), we propose **heterogeneous graph attention networks to model different types of entities**. Our extensive experiments and systematical analysis demonstrate that LinKG can achieve linking accuracy with an F1-score of 0.9510, significantly outperforming the state-of-the-art. LinKG has been deployed to Microsoft Academic Search and AMiner to integrate the two large graphs. We have published the linked results---the Open Academic Graph (OAG)\footnote\urlhttps://www.openacademic.ai/oag/ , making it the largest publicly available heterogeneous academic graph to date.

## 1. Introduction
* Entity linking (ontology alignment, disambiguation)
  * task of determining the identity of entities across different sources
  * entity matching, entity resolution, web appearances disambiguation, name identification, object distinction, name disambiguation과 연관되어있음
  * Web-scale의 heterogeneous entity graph를 다른 소스로부터 받아서 연결시키는 것은 아직까지 어려운 문제임
    * Web-based entity graphs are heterogeneous, as they consist of various types of entities
    * entity graph에 ambiguous entity가 많음 (ex. James Smith라는 사람이 10,000이 넘음)

* 대규모 heterogeneous graph를 연결시키기 위해 two academic entity graphs (MAG, AMiner)을 연결해야 하며, entity에 대한 ambiguity 뿐만 아니라 그래프 별로 존재하는 다른 속성들도 고려해야 한다.
  ![image](https://user-images.githubusercontent.com/60350933/153822749-723e3594-b1de-4b02-a21f-18aa4732e2ab.png)
  * 선행연구
    * 가장 명확한 방법은 heuristic rule을 사용하여 entity alignment를 만드는 것이지만, 다른 데이터에 적용하기엔 어렵다
    * alignments via learning algorithms such as neural networks and probablistic frameworks -> complexity가 높으며 규모가 큰 그래프를 다루기 어렵다
    * human annotators into loop of entity linking -> 규모가 큰 그래프를 다루기 어려움
  * **LinKG** 제안
    * three linking module로 구성되어 있고, 각각은 venue, paper, author에 대한 entity에 매칭이 됨
      * venue (coarse-grained, word-sequence dependent entity): long short-term memory networks (LSTM)를 사용하여 venue 이름의 연속적 dependency를 포착
      * paper (relatively less ambiguous and large scale): locality-sensitive hashing and CNN
      * author (highly ambiguous): heterogeneous graph attention networks (HGAT)
      * *finegrained* and *coarse-grained classification*?
         * Coarse-grained classification에 비해 상대적으로 finegrained classification은 비슷한 특징을 가진 class를 분류하는 task이다. 
         [Fine-grained classification]
         ![image](https://user-images.githubusercontent.com/60350933/153830038-b83a96f4-94e2-497f-a896-af241761ebce.png)

    * HGAT 전 step에서 linked venue와 paper entities를 넣는다. 

* Contribution
  * 두개의 대규모 graph를 연결시킴
  * linking result를 기반으로 OAG (open academic graph)를 출판함 - largest public academic data 

## 2. Related Work
* Entity linking
  * rule based methods and develop a rule discovery 알고리즘
  * 머신러닝 기반
    * entity matching은 의사결정에서 Bayesian risk를 최소화하는 것
    * labeled data를 줄여서 semi-supervised or unsupervised matching algorithm (factor graph model, pairwise similarity 등)을 도입
    * COSNET: energy-based model (multiple network의 global과 local consistency를 고려)
  * network embedding 기반 접근법
    * network 간의 user embedding을 동일하게 학습하는 것을 최적화하는 framework
    * MEgo2Vec: attention mechanism과 graph neural network를 사용하여 two candidate user의 ego network를 통합

* 한계: there is no unified solution for large-scale heterogeneous entity linking
  * hashing-based method to efficiently find linkings with less ambiguity. 
  * LSTM to perform fuzzy-sequence linking
  * CNN to perform fine-grained text matching
  * 마지막으로 graph attention network 사용

## 3. Problem definition
#### Def 3.1. Heterogenous entity graph (HEG)
 HEG는 *e*∈E, *r*∈R이 type mapping function r(*e*): E→ C,ϕ(r): R→D와 연관된 graph *HG*={E,R} 이다. 이때 C와 D는 entity와 relation type을 가진다. 
 * ex. academic graph에서 entity는 author, paper, venue로 구성되어 있고, relation set D는 authorship (author-paper), paper publishes in venue (paper-venue), author publishes in venue (author-venue), co-authorship (author-author)로 나뉜다. 

#### problem 3.1. Entity linking across HEGs
* HG1, HG2가 존재할 때, entity linking L={(e1,e1)|e1∈ HG1, e2∈ HG2}를 생성하는 것
* 해당 연구에서는 MAG와 AMiner를 사용할 것. 

## 4. The *LINKG* 프레임워크
![image](https://user-images.githubusercontent.com/60350933/153831229-c3a1a6c1-450c-432d-887e-2f0c35c935fb.png)
* 주요 구성
  * (1) Venue Linking
     * 각 graph의 venue의 full name이 주어졌을 때, 각 그래프의 동일한 venue를 연결 시키는 것으로, venue full name만 사용하는 것이 심플하고 효율적이며 효과적이다. 
     * **venue name matching**과 **LSTM을 통한 sequence embedding**으로 구성되어 짐
  * (2) Paper Linking
     * Heterogeneous 정보 활용 (paper의 title, publication year, authors, publication venue)
     * 각 데이터 별로 수만개의 정보가 존재하기 때문에 1) hashing technique (locally-sensitive hashing)과 2) effective linking을 위한 CNN을 제안
  * (3) Author Linking
     * 각 author 별로 heterogeneous subgraph를 생성
       * 하나의 subgraph는 coauthor, paper, publication venue로 구성되어져 있음
     * venue와 paper linking result를 author linking에 포함
     * author entity의 ambiguity를 해결하기 위해 HGAT를 도입함
 * 각 entity의 type이 다른 속성을 가지고 있기 때문에 각각에 대한 challenge를 해결하기 위해 3개의 다른 neural network를 구성함

#### 4.1. Linking venues- sequence based entities
* **Word-sequence dependent entity**
  *  venue의 경우 venue의 속성인 keyword, publication, author로 matching을 시키면 되지만, 각 venue 별로 그 데이터의 수가 많기 때문에 이러한 정보를 바로 사용하는 것이 용이하지는 않으며, keyword를 사용하기에는 유사한 분야 내의 venue들은 유사한 키워드를 가지기 때문에 구별하기가 용이하지 않다. 
* Name matching
  * full name과 abbreviation 을 사용한 direct matching은 27,000개의 venue pair, 즉 실제 matching과 유사한 결과를 낳지만, 그 외의 venue들은 직접적으로 이어질 수 없으므로 이를 해결하기 위한 방법들을 다음과 같이 사용함
    * Word order inversion: two graph 간의 동일한 저널이 다른 word order로 나타나는 경우 (Journal of Shenzhen university & Shenzhen University Journal)
    * Extra or missing prefix or suffix: additional annotation이 존재하는 경우 (ex. Proceedings of the Second international conference~~)
* LSTM
  * 완벽히 매치가 되지 않는 venue를 다루기 위해 full name에 존재하는 relative word나 keyword sequence를 주목함. 
    * **Integral Sequence**
      * venue name에 존재하는 raw word sequence 
    * **Keyword Sequence**
      * Integral sequence에 존재하는 키워드 
  * 계산 과정
    * 1) LSTM layer로 venue i에 대한 integral sequence와 keyword sequence 계산
    * 2) 다른 그래프 간의 차이를 계산
    ![image](https://user-images.githubusercontent.com/60350933/153837810-f0996350-d9b8-4091-80b9-d58de86cf5c8.png)
    * 3) 자카드 인덱스와 keyword sequence의 inversion pair의 number을 다 합친 후, 2개의 venue에 대한 similarity를 게산하기 위해 fully-connected layer 도입
      ![image](https://user-images.githubusercontent.com/60350933/153837954-a4d03ecb-fdc3-4f9b-8ea3-8b9c0ade39e7.png)
  * 학습을 위해 labeled candidate를 사용함

#### 4.2. Linking Papers- Large-scale entities
* Challenges
  * 1) Large volume of academic publications
  * 2) Often truncated titles such as '?' and ':'
  * 3) 다른 연구이지만 동일한 제목, 동일한 venue에서 출판된 연구가 존재
* Locally-sensitive hashing
  * 모든 possible pair를 다 고려한다면 O(n^2) 만큼의 complexity가 발생됨 (n=number of papers)
  * locally sensitive hashing을 통해 nearest neighbor을 효율적으로 탐색할 수 있다. 
  * map titles to binary codes
    * one-hot encoding을 사용할 수 있지만 binary representation은 high-dimension을 가질 것이고 LSH로 mapping을 하는 것은 정보 손실의 위험이 있다. 
    * (해결책) Doc2Vec을 사용하여 title을 real-valued vector로 변환
  * LSH를 사용하여 real-valued vector을 binary code로 변환  
* CNN
  * LSH를 통한 정보 손실은 paper candidate의 link 손실로 이어짐
  * 이러한 unlinked paper에 대한 fine-grained matching signal을 파악하기 위해서 CNN을 사용
  * CNN based linking strategy
    * 1) candidate pair research 
      * title keyword 기반의 inverted index technique을 사용하여 candidate paper pair filtering 진행  
      * *inverted index*는 content와 document 내의 위치를 기록하는 index data structure로, hashmap과 같은 역할을 한다. 
         [이미지 출처 (2)]
         ![image](https://user-images.githubusercontent.com/60350933/153843265-70130648-bff6-47fa-8302-012c88307a07.png)
    * 2) paper similarity matrix construction
      * 각 후보 candidate 별로 two similarity matrix를 구축하여 CNN의 input으로 사용
      * similarity matrix의 z(ij)는 paper의 제목이나 author의 i 번째 word (name initial)과 j번째 word (name initial)이 동일하면 1로, 아니면 -1로 표시가 된다. 
      * paper의 제목과 author에 대한 각각의 similarity matrix가 구축이 되어 CNN에 사용됨. 
    * 3) CNn-based pairwise similarity learning
      ![image](https://user-images.githubusercontent.com/60350933/153846199-4e5cde11-413f-4922-8bff-4b69dbb813ea.png)
      * CNN의 첫 layer에서 n번쨰 filter는 input similarity matrix를 통해 feature map을 생성함
      * 이후 square filter와 ReLU가 적용이 되며, data heterogeneity에 수반되는 다양한 similarity pattern을 파악하기 위해 multiple filter이 사용됨
      * 이후 따라오는 layer는 convolutional or pooling layer이며, higher-order matching feature을 capture하기 위해 사용이 됨
      * hidden layer matrix를 dense vector로 flatten한 후에 title과 author similarity matrices로 부터 추출된 2개의 벡터를 concategating함
      * MLP를 통해 final matching score을 생산
      * Output을 도출하기 위해 softmax function이 사용이 됨

#### 4.3. Linking authors - ambiguous entities
* 요약
  * Generate candidate pairs for authors
  * Candidate pair-> construct a heterogeneous ego subgraph and two ego subgraph는 그들이 공동 venue나 paper를 공유하면 connected 될 수 있다.  
  * author matching을 위한 HGAT
* Paired subgraph construction
  * Candidate pair 별로 direct neighbor이 선택이 됨 (이 때 coauthor, paper, venue가 선택) 
    ![image](https://user-images.githubusercontent.com/60350933/153847476-12d1293a-97b5-4f68-befb-9288ad9c56e8.png)
    * 두 명의 author이 connected 된 적이 있다면 그 2개의 subgraph를 하나로 합칠 수 있음
    * collaboratioin과 publication frequencies를 고려하여 fix-size paired subgraph를 구축하는 것
    * Coauthors' paper과 venue를 paired subgraph를 통해 construct하는 것.  
    * pair graph를 생성하기 위해 coauthor의 paper과 venue 또한 고려 (two-hop ego graph)
* HGAT
  ![image](https://user-images.githubusercontent.com/60350933/153849968-cb2932ac-c833-4708-b316-5043b10796a6.png)

  * neighbor로부터 파생되는 정보를 합치기 위해 GAT를 사용
  * semantic and structure information을 사용하여 각각의 embedding을 concatenated하여 input feature으로 만든 후에 GAT를 pre-train함
    * semantic feature: skip-gram word embedding model을 AMiner의 publication corpus (title, author, abstract)에 적용하여 생성한 후, 각 entity에 대한 semantic entity를 연관된 단어의 embedding의 평균을 통해서 도출함
    * structure features: LINE model을 사용  
  * Encoding layer
    * Multiple graph attention layer로 구성
    * source entity가 target entity에 가지는 aggregation weight을 의미하는 attention coefficient attn을 학습하기 위해 사용. 
    * Attention coefficient는 self-attention mechanism을 통해서 학습이 됨
       ![image](https://user-images.githubusercontent.com/60350933/153849531-b84e9470-c756-4b3c-bf1a-3a13cfb88aa9.png)
    * subgraph 구조를 활용하여 graph attention layer은 식(5)만 계산을 하면 되고, 가능한 모든 ej에 대해서 o는 softmax function을 통해 정규화가 된다
       ![image](https://user-images.githubusercontent.com/60350933/153849666-8c880786-2ae3-4c8e-8de1-eee1bdbaf7dd.png)
    * 기존의 GAT와 달리 각 type별 entity에 대해 다른 attention parameter를 사용하는데 이는 author linking에 있어서 각 entity가 다른 역할을 하기 때문이다. 
    * node e*i*에 대한 output embedding h를 generate 하기 위해 multi-head attention을 계산한다.
      ![image](https://user-images.githubusercontent.com/60350933/153849924-baaad85b-6759-4873-8bc2-a02e460a1b9c.png)
    * 총 2개의 attention layer을 사용한다.
  * Output layer
    * Graph encoder을 통과한 후에 near neighbor을 aggregate함을 통해 각 node는 hidden embedding을 보유하게 된다. 
    * candidate pair의 2 focal authord에 대한 embedding을 하나의 vector로 합친 후, 2개의 fully-connected layer을 사용하여 각 pair 별 output representation을 생성한다. 
    *  마지막으로, negative log-likelihood function을 optimize

#### 4.4. Further discussions
* LSH와 CNN의 combination을 활용해볼만함

## 5.Result
![image](https://user-images.githubusercontent.com/60350933/153850923-931ca490-8045-4ea3-8c2b-c1a8a780688f.png)

 ## 출처
(1) https://towardsdatascience.com/dog-breed-classification-hands-on-approach-b5e4f88c333e
(2) https://www.geeksforgeeks.org/inverted-index/

