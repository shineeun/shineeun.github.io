---
layout: post
title:  "[논문리뷰] Modeling Relational Data with Graph Convolutional Networks "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Schlichtkrull, M., Kipf, T. N., Bloem, P., Berg, R. V. D., Titov, I., & Welling, M. (2018, June). Modeling relational data with graph convolutional networks. In European semantic web conference (pp. 593-607). Springer, Cham.

Knowledge graphs enable a wide variety of applications, including question answering and information retrieval. Despite the great effort invested in their creation and maintenance, even the largest (e.g., Yago, DBPedia or Wikidata) remain incomplete. We introduce **Relational Graph Convolutional Networks (R-GCNs)** and apply them to two standard knowledge base completion tasks: Link prediction (recovery of missing facts, i.e. subject-predicate-object triples) and entity classification (recovery of missing entity attributes). R-GCNs are related to a recent class of **neural networks operating on graphs**, and are developed specifically to deal with the **highly multi-relational data characteristic of realistic knowledge bases**. We demonstrate the effectiveness of R-GCNs as a stand-alone model for entity classification. We further show that factorization models for link prediction such as DistMult can be significantly improved by enriching them with an encoder model to accumulate evidence over multiple inference steps in the relational graph, demonstrating a large improvement of 29.8% on FB15k-237 over a decoder-only baseline.

## 1. Introduction 
* statistical relational learning (SRL)에서 knowledge bases의 missing information을 에측하는 것이 가장 중요함. 
* Knowledge bases는 directed labeled mutigraph로, entity가 node가 되며, triples는 labeled edges로 encoded된 것이다. 
  ![image](https://user-images.githubusercontent.com/60350933/153741077-fe34a0fb-af73-44f8-adfc-2fb001e30dd6.png)

* SRL의 task
  * ㅣ 
