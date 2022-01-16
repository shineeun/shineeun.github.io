---
layout: post
title:  "[논문리뷰] A survey on knowledge graphs: representation, acquisition and application "
subtitle:  "Knowledge Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Ji, S., Pan, S., Cambria, E., Marttinen, P., & Philip, S. Y. (2021). A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on Neural Networks and Learning Systems.

## 1. Overview
### A. Brief history of knowledge bases
   * Reasoning and problem solving에 많이 사용되어 왔으며, resource description framework (RDF)와 Web Ontology Language (OWL)을 통해서 Semantic web에 대한 중요한 기준들이 세워졌다. 
   * 이를 기반으로, WordNet, DBpedia, YAGO, Freebase와 같은 knowledge bases와 ontologies가 published 됨
   * Knowledge Vault를 기반으로 한 Google의 search의 출시로 knowledge graph의 개념이 널리 알려졌다. 

### B. Definitions
   * **Knowledge graph**
     * ontology로부터 정보를 얻고 ontology에 정보를 통합하며, 새로운 지식을 얻기 위해 reasoner을 apply하는 것
     * Entity(node)와 relation(different types of edges)으로 구성된 multi-relational graph. 
     * G = {E, R, F}
       * E= entities, R= relations, F=facts
       * (h, r, t) ∈ F
       
   * TABLE 1. Notations and descriptions
     ![image](https://user-images.githubusercontent.com/60350933/149667735-25e3bb91-02d4-438c-ae15-366594a5e1c3.png)
   
### C. Knowledge graph 연구 분류
   * Figures 2. Categorization of research on knowledge graphs
     ![image](https://user-images.githubusercontent.com/60350933/149667760-bb34abae-f7e5-4cd3-8bc4-884823a3a3be.png)

#### 1) Knowledge Representation Learning
Representation learning은 다음과 같은 4가지의 측면으로 볼 수 있다. 
* **Representation space** : relations과 entities가 represented됨
  - point-wise space, manifold, complex vector space, Gaussian distribution, and discrete space
* **Scoring Function** : factual triples의 plausibility를 측정
  - distance-based와 scoring function을 기반으로한 similarity matching으로 나뉨. 
* **Encoding models** : relational interaction을 representing하고 learning
  - 최근 연구가 많이 집중되고 있는 분야
  - linear/bilinear models, factorization, neural network 등이 있다. 
* **Auxiliary information** : 외부 정보 활용
  - textual, visual and type information을 고려함 
  
#### 2) Knowledge Acquisition
* **KGC**
  - expanding existing knowledge graphs 
  - 주요 카테고리: embedded-based ranking, relation path reasoning, rule-based reasoning, and meta relational reasoning
* **relation extraction**
  - discover new knowledge from the text
  - 주요 카테고리: recognition, disambiguation, typing, and alignment
  - attention mechanism, graph convolutional networks (GCN), adversarial training, reinforcement learning, deep residual learning, transfer learning 등을 주로 사용함.
* **entity discovery**
  - discover new knowledge from the text

#### 3) Temporal knowledge graphs
* Temporal 정보를 representation learning에 사용함
* 주요 카테고리: Temporal embedding, entity dynamics, temporal relational dependency, temporal logical reasoning

#### 4) Knowledge-aware Applications
* 주요 카테고리: Natural language understanding, Question answering (QA), recommendation system 등

## Knowledge representation learning (KRL, KGE, multi-relation learning, statistical relational learning)
KRL 모델을 학습하는 전략은 Appendix D에 소개됨
### A. Representation space
### B. Scoring function
### C. Encoding models
### D. ausiliary information



