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

## Abstract
Human knowledge provides a formal understanding of the world. Knowledge graphs that represent structuralrelations between entities have become an increasingly popular research direction towards cognition and human-level intelligence. In this survey, we provide a comprehensive review of knowledge graph covering overall research topics about 1) knowledge graph representation learning, 2) knowledge acquisition and completion, 3) temporal knowledge graph, and 4) knowledge-aware applications, and summarize recent breakthroughs and perspective directions to facilitate future research. We propose a full-view categorization and new taxonomies on these topics. Knowledge graph embedding is organized from four aspects of representation space, scoring function, encoding models, and auxiliary information. For knowledge acquisition, especially knowledge graph completion, embedding methods, path inference, and logical rule reasoning, are reviewed. We further explore several emerging topics, including meta relational learning, commonsense reasoning, and temporal knowledge graphs. To facilitate future research on knowledge graphs, we also provide a curated collection of datasets and open-source libraries on different tasks. In the end, we have a thorough outlook on several promising research directions.

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
Key Issue: to learn low-dimensional distributed embedding of entities and relations.
* Embedding space should follow three conditions: differentiability, calculation possibility, and definability of a scoring function. 
#### 1) Point-wise space
* TransE: represents entities and relations in d-dimension vector space and makes embedding follow the translational princible
* TransR: introduces separated spaces for entities and relations. 
  * Entities를 relation으로 projection matrix(Mr ∈ Rk×d)를 통해서 project 
* NTN: models entities across multiple dimensions by a bilinear tensor neural layer. The relational interaction between head and tail is captured as tensor
* HAKE: captures semantic hierarchies by mapping entities into polar coordinate system
#### 2) Complex vector space
e h, t, r ∈ C_d
head entity= Re(h))+iIm(h) whree Re(h) is the real part and IM(h) is the imaginary part
* ComplEx, RotatE
### B. Scoring function
### C. Encoding models
### D. ausiliary information



