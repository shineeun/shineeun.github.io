---
layout: post
title:  "[논문리뷰] TComplEx: Tensor Decompositions For Temporal Knowledge Base Completion "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Lacroix, T., Obozinski, G., & Usunier, N. (2020). Tensor decompositions for temporal knowledge base completion. arXiv preprint arXiv:2004.04926.

## Abstract
Most algorithms for representation learning and link prediction in relational data have been designed for static data. However, the data they are applied to usually evolves with time, such as friend graphs in social networks or user interactions with items in recommender systems. This is also the case for knowledge bases, which contain facts such as (US, has president, B. Obama, [2009-2017]) that are valid only at certain points in time. For the problem of link prediction under temporal constraints, i.e., answering queries such as (US, has president, ?, 2012), we propose a solution inspired by the canonical decomposition of tensors of order 4. We introduce new regularization schemes and present an extension of ComplEx (Trouillon et al., 2016) that achieves state-of-the-art performance. Additionally, we propose a new dataset for knowledge base completion constructed from Wikidata, larger than previous benchmarks by an order of magnitude, as a new reference for evaluating temporal and non-temporal link prediction methods.

## 1. Introduction
* Background
  * Relational data에서의 link-prediction은 추천시스템, knowledge base completion 등 다양한 분야에 적용되어 왔으며, relational data는 주로 시간에 의해 변화한다(Temporal).
  * **Temporal link prediction**은 특정 시간대에서 missing link를 찾는 것을 목적으로 한다. 
* 연구주제 
  * **Temporal link prediction을 temporal knowledge base completion의 관점으로 접근 **
    * Knowledge base에서의 link prediction은 (subject, predicate,?)에서 potential objects의 accurate ranking을 제공함으로, incomplete한 query에 답하고자 한다. 
    * 이 때 knowledge bases에는 temporal metadata가 포함된다. 
    * 따라서, (subject, predicate, ?, timestamp)의 quadriple에서 '?'에 대한 답이 timestamp에 따라 달라질 것이다. 
  * 4 tensor completion problem으로 문제를 해결하고자 함
    * Tensor factorization methods가 Knowledge Base Completion에서 우수한 성능을 보임 (Nickel et al., 2016a; Trouillon et al., 2016; Lacroix et al., 2018). 
    * Time stamps는 4-th mode in the binary tensor holding (subject, predicate, object, timestamps) facts를 구별하고 index하는데 사용된다. 
  * ComplEx (Troullion et al., 2016) decomposition of this order 4 tensor
    * The decomposition은 각 timestamp 별로 embedding을 만든다. 
    * Natural Prior: timestamp representation이 시간에 따라 변화할 수 있도록 하며, 이 prior은 nuclear p-norm의 변화의 optimum 값인 regularizer로 볼 수 있다. 
      * Nuclear norm은 sum of singluar values 로, 주로 rank minimization problems의 convex heuristics에 사용된다. 

## 2. Related work
#### 2.1. Static link prediction
* standard tensor decomposition methods가 좋은 성능을 가짐
  * Canonical Polyadic (CP) Decomposition: tensor equivalent to the low-rank decomposition of a matrix
    * A tensor X of canonical rank R
      (수식 넣기, page 2)
  * Dismult
  * Use of complex parameters and set W to the complex conjugate of U,U_bar
* 이러한 알고리즘을 tensor nuclear norm의 다양한 form으로 regularizing하거나 learning objective를 변경하는 것은 SOTA로 이어짐

#### 2.3. Temporal link prediction 방법론
* 다양한 선행연구
  * Bayesian과 learning method for representing temporal relations.
    * 위 사용한 temporal smoothness와 본 연구에서 사용하는 gradient penalty가 유사함 
  * ASALSAN: tensor decomposition to express stemporal relations
* Temporal Knowledge base completion
  * Goel et al. (2020) learns entity embeddings that change over time, by masking a fraction of the embedding weights with an activation function of learned frequencies
  * ConT (Ma et al., 2018): 각 timestamp별로 새로운 core tensor을 학습한다
  * Garcia-Duran et al. (2018): time dimension을 에측해야 할 sequence로 간주하여, recurrent neural network (RNN)을 사용하여 embedding을 transform하였다. 
* Lacroix et al. (2018)의 canonical tensor decomposition을 확장하여 training set을 order 4 tensor로 표현한다. 

## 3. Model - TComplEx
* (subject, predicate, object)이 timestamp와 함께 제공된다. 
* timestamp range를 구분하여, trainingset을 4-tuple (subject, predicate, object, time)인 order 4 tensor indexing이 가능하도록 한다. 
* Loss
  ![image](https://user-images.githubusercontent.com/60350933/151537014-7a78e108-fdc8-4edb-b2d2-2cb27ac640e0.png)
  * 각 train tuple (i,j,k,l)에 대하여 instantaneous multiclass loss를 최소화한다. 
  * 해당 loss는 (subject, predicate, ?, time)의 형태에 대한 query에만 적용이 된다. 
* Objective function with weighted regularizer Ω
  ![image](https://user-images.githubusercontent.com/60350933/151537036-491cb2d0-c14a-438a-9872-6d136ba2964a.png)
* Decomposition
  * TComplEx: New factor T를 추가하여 ComplEx decomposition 확장 
    ![image](https://user-images.githubusercontent.com/60350933/151537074-e162aa35-b7e4-4208-876a-8ee3f1e2ffff.png)
  * multi-linear dot product를 모듈화하는 timestamp embedding이 추가됨 
  * time-dependent representation을 얻기 위해 timestamp를 object, predicates, subject를 동일하게 모듈화 하는데 사용될 수 있다. 
    ![image](https://user-images.githubusercontent.com/60350933/151537120-16a75a0e-51bf-467c-9cfc-4e1b7150d19f.png)

#### 3.1. Non-Temporal predicates
* Tensor X_hat 
  * heterogenous knowledge base에는, timestamp에 의해 영향을 받지 않는 predicates가 존재할 수 있다. 
  * Tensor X_hat를 temporal, non-temporal한 두개의 tensor의 합으로 decompose할 수 있다. 
    ![image](https://user-images.githubusercontent.com/60350933/151537161-a64bfd29-5409-4af4-af62-a240a26e9c13.png)
  * 이러한 decomposition을 **TNTComplEx**. 
  * Temporal과 non-temporal한 tensor들 간의 parameter sharing을 통해, hyperparameter을 하나 줄일 수 있다. 

#### 3.2.Regularization
* order 4 tensor은 order 3 tensor에 unfolding modes가 합쳐진 것으로 볼 수 있다. 
* Decomposition (2)와 (3) 모두 order 3 tensors에 temporal과 predicates mode를 unfolding하지 않은 것으로 볼 수 있다. 
* 따라서, 이러한 unfolding으로 부터 도출될 수 있는 decomposition을 고려했을 때의 weighted regularizer는 다음과 같이 정의될 수 있다. 
  ![image](https://user-images.githubusercontent.com/60350933/151537252-99864745-fba1-45ee-b03b-b4671c3eec50.png)
  * The first regularizer는 object, predicates, 그리고 (preidicate, timestamp) pair을 그들의 각자 marginal probabilities에 따라 가중치를 둔다. 
    * 해당 regularizer은 weighted nuclear 3-norm on an order 4 tensor의 variational form이라고 볼 수 있다. 
  * The secon regulairzer은 nuclear 3가 2개의 tensors (tensor1추가) 와 (tensor2추가)에 준 pnealty의 합과 같다.

#### 3.3. Smoothness of temporal embeddings
* priori structure on the temporal mode
  * 즉, 이웃하는 timestamps 간의 close representation을 가질 것으로 기대할 수 있다. 
  * **Penalize the norm of the discrete derivative of the temporal embeddings**
    ![image](https://user-images.githubusercontent.com/60350933/151537295-0c4885d9-e880-42b3-bc52-46053bf18dc9.png)
  * the sum of 식5와 variational form of the p norm은 new tensor atomic norm의 variation 형태로 볼 수 있다. 

#### 3.4. Nuclear p-norms of tensors and their variational forms
* regularizers로 tensor nuclear p-norms 사용함. 
  * The nuclear p-norm of tensor of order D
    ![image](https://user-images.githubusercontent.com/60350933/151537327-73b73c26-602f-4f5b-8bd8-a81a23cd9647.png)
  * nulear p-norm은 tensor을 unit p-norm factor의 rank-1 tensor인 atom의 합으로 본다. 
  * NP-hhard to compute하기 때문에 variational form을 사용함
    ![image](https://user-images.githubusercontent.com/60350933/151537367-ff698d65-974e-4f8f-93fa-21f2dc078bf7.png)
    * 이때 equality가 성립하기 위해서 모든 가능한 R에 대해서 infimum이 이루어져야 하고, docomposition의 desired rank로 R을 고정하는 것이 practical하다. 
* 해당 regularizer은 timestamp와 predicate modes의 unfolding을 고려하여, marginal의 product가 아닌 timestam와 predicate의 joint marginal에 따른 weight을 구할 수 있게 된다.
  * 이는 timestamp와 predicate가 independent하지 않을 때 중요하게 작용함 

#### 3.5. Experimental impact on the regularizers
* ICEWS05-15 dataset에 활용

## 4. A new dataset for temporal and non-temporal knowledge base completion
* Wikidata의 status quo
  * temporal information is specified in the form of 'OccurSince' and 'OccurUntil' appended to triples
* Wikidata의 수정
  * (subject, predicate, object)와 timestamp (begin, end)로 구성

## 5. Experimental results
#### 5.2. Results
![image](https://user-images.githubusercontent.com/60350933/151537450-01f5ab74-c254-4a36-83a0-5c3d70414e8d.png)
  * performance that are stable through a tenfold inrease of its number of parameters in static dataset
  * performance increases a lot with the numer of parameters with static dataset

## 6. Qualitative study
* Cross-entropy loss along the temporal tubes
  * to enforce a stronger temporal consistency, and be able to answer the queries of the type (subject, predicate, object, ?)
  ![image](https://user-images.githubusercontent.com/60350933/151540203-12d5f325-9883-4201-813b-d158b6508272.png)
  * Optimize the sum of *l* in queation 7 and *l*  to better answer the queries along the time axis
* The models are able to learn rankings that are correct along time intervals despite our training method only ever sampling timestamps within these intervals
  ![image](https://user-images.githubusercontent.com/60350933/151540457-4cfff337-f0b9-4088-a3e8-f07376510e0e.png)

## 7. Conclusion
* point-in-time, beggining and endings, interval 관련한 dataset에 잘 적용이 됨
* low-dimensional model과 high-dimensional model의 gap을 매꿔줌 
* large scale temporal dataset 제안함

## References
* Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. Diachronic embedding
for temporal knowledge graph completion. In AAAI, 2020.
* Alberto García-Durán, Sebastijan Dumanciˇ c, and Mathias Niepert. Learning sequence encoders for ´
temporal knowledge graph completion. arXiv preprint arXiv:1809.03202, 2018
* Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. Canonical tensor decomposition for
knowledge base completion. In Proceedings of the 35th International Conference on Machine
Learning (ICML-18), pp. 2863–2872, 2018.
* Yunpu Ma, Volker Tresp, and Erik A Daxberger. Embedding models for episodic knowledge graphs.
Journal of Web Semantics, pp. 100490, 2018.
* Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International Conference on Machine Learning,
pp. 2071–2080, 2016.

