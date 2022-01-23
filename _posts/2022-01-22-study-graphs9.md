---
layout: post
title:  "[논문리뷰] Knowledge-aware Path Recurrent Network (KPRN) - Explainable reasoning over knowledge graphs for recommendation "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Wang, X., Wang, D., Xu, C., He, X., Cao, Y., & Chua, T. S. (2019, July). Explainable reasoning over knowledge graphs for recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 5329-5336).

## Abstract
Incorporating knowledge graph into recommender systems has attracted increasing attention in recent years. By exploring the interlinks within a knowledge graph, the connectivity between users and items can be discovered as paths, which provide rich and complementary information to user-item interactions. Such connectivity not only reveals the semantics of entities and relations, but also helps to comprehend a user’s interest. However, existing efforts have not fully explored this connectivity to infer user preferences, especially in terms of modeling the sequential dependencies within and holistic semantics of a path.
In this paper, we contribute a new model named Knowledgeaware Path Recurrent Network (KPRN) to exploit knowledge graph for recommendation. KPRN can generate path representations by composing the semantics of both entities and relations. By leveraging the sequential dependencies within a path, we allow effective reasoning on paths to infer the underlying rationale of a user-item interaction. Furthermore, we design a new weighted pooling operation to discriminate the strengths of different paths in connecting a user with an item, endowing our model with a certain level of explainability. We conduct extensive experiments on two datasets about movie and music, demonstrating significant improvements over state-of-the-art solutions Collaborative Knowledge Base Embedding and Neural Factorization Machine.

## 1. Introduction
* Knowledge Graph (KG) 
  * Comprehensive auxiliary data: background knowledge of items and their relations amongst them. 
    * Organizes facts of items in triplets (H,R,T) which can be integrated with user-item interactions   
      ex. (Ed Sheeran, IsSingerOf, Shape ofYou)
  * Interlinks를 탐색함을 통해, user와 item 간의 연결성이 그들의 underlying 관계를 드러내며, 이는 user-item interaction data를 보조하는 용도로 사용이 된다. 
  * Extra user-item connectivity 정보는 recommender system에서 reasoning과 explainability이 가능하게 함. 
    ![image](https://user-images.githubusercontent.com/60350933/150627462-09e0a7d1-3940-4fec-909e-9e4a8e587459.png)
    * By syntehnsizing information from paths, connectivity helps to reason about unseen user-item interactions (potential recommendations)
      * reasoning은 recommendation의 behind reason을 제공한다. 

* Research Gap
  Knowledge graph 기반 추천시스템은 1) Path-based methods와 2)Embedding-based methods로 나뉜다. 
  * 1) Path-based methods: refine similarities between users and items
    * Relations are excluded from meta-paths: thus when similar entities but different relations are involved, it can not be specified. 
    * Fail to uncover and reason on unseen connectivity patterns
    * Meta-path requires domain knowledge to be predefined
  * 2) Knowledge graph embedding (KGE) techniques to regularize the representation of items
    * TransE, TransR 
    * Items with similar connected entities have similar representations, which faciliate the collaborative learning of user interests
    * 하지만, reasoning ability는 부족하고, entity간 직접적인 관계만 고려할 수 있고, user-item connectivity의 특징화는 implicit하게 진행된다 (user의 선호도를 infer하는 것이 아닌 representation learning을 잘하기 위함)
  
* **Knowledge-aware Path Recurrent Network (KPRN)**
  * Solution that reasons on path to infer user preferences on items
  * Model the **sequential dependencies** of entities and relations of path connecting user-item pair.
  * Steps
    * 1) Extract qualified paths between a user-item pair from the KG
    * 2) Adopt LSTM network to model the sequential dependencies of entities and relationships
    * 3) Pooling operation: **aggregate** the representations of paths to obtain predictionn signal for the user-item pair and **discriminate** the contributions of different paths for a prediction

## 2. Knowledge-aware Path Recurrent Network
#### 2.1. Background
* KG= {(h,r,t)|h,t ∈E, r∈R}, where each triplet (h,r,t): relationship r from entity h to tail entity t. 
* Bipartite graph of user-item interaction data
  * user_set ![image](https://user-images.githubusercontent.com/60350933/150631346-f4043ed2-4ea0-447d-8922-c5c4930b650b.png)
  * item_set ![image](https://user-images.githubusercontent.com/60350933/150631361-9bf99922-6c02-4633-ae93-226dba4fe172.png)
  * user-item interaction tripliet: τ=(u,interact,i)
    * interact: pre-defined relation. 
* Merge item set and entity set 
  * I ⊂ E, so that two data are integrated into G={(h,r,t)|h,r∈E', r∈R'}, where E'=E∪U and R'=R∪{interact}
* KG: combined graph G including both original KG and user-item data. 

#### 2.2. Preference Inference via Paths
* Within G, the path from user *u* to the item *i* is defined as a sequence of entities and and relations
* Model the high-level semantics of path compositionally by considering both entities and multiple-step relations.
  * Examples: same item에서 시작하여 same item으로 끝나는 경우에도 그 사이에 다양한 구성적인 semantics와 listen behavior이 존재함을 알 수 있다. 
    * p1=[Alice -> Shape of You -> Divide -> Castle on the Hill]
    * p2=[Alice -> Shape of You -> Ed Sheeran -> Castle on the Hill]
    * p3=[Alice -> Shape of You -> Tony -> Castle on the Hill]
  * From view of reasoning: connectivity along all paths to learn compositional relation representations, and weighted pool them together for predicting the interact relationship between user and the target item. 
* Task formulation 
  * given user *u*, target item *i*, and set of paths *P*(*u*,*i*)={p1, p2, ..., pk} connecting *u* and *i*, 
    estimate the predicted score for the user-item interaction :
    ![image](https://user-images.githubusercontent.com/60350933/150632350-bac927e3-dfbe-486a-854b-46f7739cfae7.png)

#### 2.3. Modeling
* KPRN takes a set of paths of each user-item pair as input and outputs a score indicating how possible the user will interact the target item
  ![image](https://user-images.githubusercontent.com/60350933/150632424-e3ac37ce-2463-4f97-a4b1-1f5c0a4179e4.png)
* Three key components: Embedding layer, LSTM layer, pooling layer
* Embedding layer: 
  * to project three types of IDs information: the entity, entity type, and the relation pointing to the next node into a latent space 
  * path *pk*가 주어졌을 때, type (ex. 사람, 영화)과 specific value (ex. 최민호, Hobbit) of each entity를 two separate embedding vector로 project
    ![image](https://user-images.githubusercontent.com/60350933/150632768-c1d68b71-cde2-4054-97df-df0881198c41.png)
    이때 *d*는 embedding 크기
  * entity-entity pair은 relationship의 차이에 따라 다른 의미를 포함하고 있으므로, relations semantics를 path representation learning에 포함한다. 
     ![image](https://user-images.githubusercontent.com/60350933/150632835-a66afa78-1306-41c3-900e-ad1ce6fbba0b.png)
  * Path *pk*에 대한 set of embeddings 형성
* LSTM layer
  * encode the elements sequentially with the goal of capturing the compositional semantics of entities conditionied on relations
  * RNN model is employed to explore the sequential imformation and generage a single representation for encoding its holistic semantics
    * **LSTM**: memorizing long-term dependency in sequence -> long-term sequential pattern은 user과 item entities를 연결하는 path를 reasoning하는 데 중요하게 작용
  *  Path step *l*-1에서 LSTM 층은 hidden state vector (consuming the subsequence [e1, r1, ..., e(l-1), r(l-1)]를 output으로 낸다. 
     ![image](https://user-images.githubusercontent.com/60350933/150646969-cf4ef8eb-4f71-4eb9-abe1-59bcdbf4503b.png)
     * Input vector은 sequential information과 entity의 semantic information, next entity와 관계가 다 포함이 됨
  * x(l-1)과 h(l-1)은 path-step의 hidden state인 l을 파악하기 위한 용도로 사용됨
    ![image](https://user-images.githubusercontent.com/60350933/150647041-bc035c3f-370b-42f3-bbc0-763df49fd8b4.png)
    이때 c*l*은 cell state vector이며, z는 information transformation module, d'는 number of hidden units이다. 
    i,o,f는 각각 input, output, forget state이다. 
    activation function은 시그모이드를 사용했다. 
    * Memory state의 장점을 사용하여, last state인 h(L)은 whole path p*k*를 나타내는 것이 가능함
  * Predict plausibility of τ: two fully-connected layers are adopted to project the final state into predictive score
     Equation 4 ![image](https://user-images.githubusercontent.com/60350933/150647169-e645d918-4705-4cb0-b57f-3ee9f919b5f7.png)
     * 이떄 W1과 W2는 coefficient weights of the first and second layers respectively
* Pooling layer: combine multiple paths and output (the final score of the given user interacting the target item)
  * S={s1,s2,...,sk} : predictive scores for K paths
  * P(u,i)={p1,p2,...,pk}: connecting a user-item pair (u,i) where each element is calculated based on Eq.4
  * **Weighted pooling operation**
    * final prediction은 모든 path의 score의 평균으로 할 수 있지만, 각 path의 중요성을 파악하지 못하기 때문
      ![image](https://user-images.githubusercontent.com/60350933/150647272-88d0239d-2514-411d-acfa-f70053e6a89a.png)
      이 때, γ는 exponential weight을 계산하기 위한 것으로 0에 가까울 수록 max-pooling이 되고, 무한대로 수렴할 수록 mean-pooling이 된다. 
  * Final prediction score
    ![image](https://user-images.githubusercontent.com/60350933/150647290-4a7017d7-b620-4fab-9472-41a22ae4393c.png)
  * Path importance를 계산하는 것이 가능해짐 (proportional to the score of each path during the back-propagation step)
    ![image](https://user-images.githubusercontent.com/60350933/150647397-81413cfc-c25d-4906-8978-556c003c127b.png)

#### 2.4. Learning
* Binary problem whether the interaction is observed
* Objective function: negative log-likelihood
  ![image](https://user-images.githubusercontent.com/60350933/150647557-a05d15dc-307b-4e80-b260-de609e163a64.png)
* L2 regularization on parameters Θ

## 3. Experiments
#### 3.1. Dataset Explanation
MovieLens-1M과 IMDb dataset (영화) KKBox(음악)

#### 3.2. Experimental settings
Evaluation metrics to evaluate the performance of top-K recommendation and preference ranking
* hit@K: relevant items are retrieved within the top K positions of the recommendation list
* ndgc@K: the relative orders maong positive and negative items within the top K of the ranking list

Baselines
* Matrix factorization with Bayesian personalied rankings (2009)
* NFM (2017)
* CKE (2016): embedding-based method
* FMG (2017): SOTA meta-path based method

#### 3.3. Performance Comparison
KPRN이 가장 좋은 성능을 보임. 

#### 3.4. Investigation of the role of path modeling
1) Effects of Relation modling 
   * Relation을 고려하지 않을 때 성능 저하
   * user과 item간의 strong connectivity가 존재할 때, path가 더 중요한 경향이 있음
2) Effects of weighted pooling
  * γ을 1에서 0.1로 줄이면, weighted pooling operation degrades the performance
    * max-pooling처럼 user-item connectivity에서 가장 중요한 path만 선택하기 때문임.  
  * γ을 1에서 10으로 늘리면 informative한 path만 고려하는 것이 아니라 더 많은 path를 aggregate하기 때문에 성능이 더 안 좋아짐 

#### 5. Conclusions
Future studies
1) mimic the propagation process of user preferences within KG vis GNNs
2) KG links multiple domains together with overlapped entities, plan to adopt zero-shot learning to solve cold start issues in the target domain. 





