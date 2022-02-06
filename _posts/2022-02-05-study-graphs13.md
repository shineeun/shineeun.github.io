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
 
 #### Capturing temporal aspect of the facts
 * 주로 timestamp나 time interval에 KG edge를 연동시킨다.
 
