---
layout: post
title:  "[논문리뷰] TeMP: Temporal Message Passing for Temporal Knowledge Graph Completion "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:
---
Wu, J., Cao, M., Cheung, J. C. K., & Hamilton, W. L. (2020). Temp: Temporal message passing for temporal knowledge graph completion. arXiv preprint arXiv:2010.03526.

## Abstract
Inferring missing facts in temporal knowledge graphs (TKGs) is a fundamental and challenging task. Previous works have approached this problem by augmenting methods for static knowledge graphs to leverage time-dependent representations. However, these methods do not explicitly leverage multi-hop structural information and temporal facts from recent time steps to enhance their predictions. Additionally, prior work does not explicitly address the temporal sparsity and variability of entity distributions in TKGs. We propose the Temporal Message Passing (TeMP) framework to address these challenges by combining graph neural networks, temporal dynamics models, data imputation and frequency-based gating techniques. Experiments on standard TKG tasks show that our approach provides substantial gains compared to the previous state of the art, achieving a 10.7% average relative improvement in Hits@10 across three standard benchmarks. Our analysis also reveals important sources of variability both within and across TKG datasets, and we introduce several simple but strong baselines that outperform the prior state of the art in certain settings.

## 1. Introduction
#### Research background
* Temporal knowledge graph(TKGs) 에서 missing fact를 추론하는 능력은 event prediction, QA, social network analysis, recommendation system에서 유용하다. 
* TKGs에서는 기존의 static KGs에서 정보를 (Obama, visit, China)와 같이 triple로 나타내는 것이 아니라, time stamp를 추가하여 (Obama, visit, China, 2014)와 같이 나타낸다. 
* 기존에는 TKGs는 static KG snapshot이 연속으로 구성된 것으로 표현되어짐 
* temporal knowledge graph completion (TKGC): 이러한 snapshot들 간의 사라진 정보를 예측하는 것

#### 선행연구
* TKGC와 관련한 연구들은 time-dependent scoring function을 발전시키는 데 집중
  * missing fact의 likelihood score을 계산
  * static KGs의 representation learning method를 기반으로 함
* **한계** 
  * (1) TKGs의 multi-hop structural information을 제대로 고려하지 않음
  * (2) KG snapshot 근처의 temporal facts를 leverage하는 능력이 없음 
    * (Obama, make agreement with, China, 2013) or (Obama, visit, China, 2012) is useful for answering the query (Obama, visit, ?, 2014)
    * *EWMA와 유사하게 최근 시점의 일에 weight를 더 주는 것인 듯*
  * (3) Temporal variability and temporal sparsity

#### Temporal variability and temporal sparsity
* Temporal variability
  * different queries를 해결하기 위해 near KG snapshot에 reference할 temporal information이 많은 것
    * political event dataset에서 (Obama, visit)에 관련한 quadruples가 (Trump, visit)에 비해 2008-2013년 사이에 더 많을 것이므로, Obama가 2014년에 _어디에_ 방문했는지에 대하 reference한 정보가 많아진다.     
* Temporal sparsity
  * 각 시간 step별로 전체 entity 중 일부만 active한 것
    * 기존 방법론에서는 **inactive entity에도 동일한 embedding을 지정**하여, time-sensitive feature을 잘 표현하지 못하였음

#### Temporal Message Passing (TeMP)
* Neural message passing + temporal dynamic models
* Temporal sparsity와 variability issue 해결 위해 Frequency-based gating + data imputation techniques 사용
* 효과
  * TKGC benchmark에서 SOTA를 달성함
  * Hits@10에서 next best model에 비해 7.3%이상의 성능 향상을 이룸
  * Fine-grained error analysis: variability를 고려하는 것이 중요함

## 2. Related Work
#### Static KG representation learning
* entity와 relation을 low-dimensional embedding으로 표현함
  * entity와 relation embedding기반으로 candidate fact에 대한 score을 계산하는 decoding method를 기반으로 하며, random negative example에 비해 valid triples가 더 높은 score을 가질 수 있도록 모델이 학습이 된다. 
  * 이러한 모델들은 entity와 관련한 multi-hop 정보를 leverage하기 위해 embedding을 생성하기 위해 shallow encoder(ex. single embedding-lookup layers)이나 GNN기반의 message passisng에 의존한다. 

#### Temporal KG representation learning
* 기존 접근법 1
  * Shallow encoder을 통해 embedding을 하고, time-sensitive quadruple decoding function을 적용함. 
    * Time-specific information은 고려되지만, event periodicity와 같은 entity-level에서의 temporal pattern은 고려되지 못함
* 기존 접근법 2
  * Message passing networks를 사용하여 intra-graph의 이웃 정보를 capture
    * temporal recurrence나 attention mechanism과 합쳐진다.
  * 1) Temporal point processes (Han et al., 2020)
    * Continuous TKGC를 파악하는 데 집중
  * 2) methods that resemble TeMP
    * Recurrent Event Networks (RE-NET) (Jin et al., 2019)
      * multi-level RNN을 사용하여 entity interaction을 modeling함 
    * DySAT (Sankar et al., 2020)
      * Self-attention을 사용하여 dynamic graph에 대한 latent node representation을 학습 
    * 이 두가지 방법은 graph extrapolation (sequence에서 next time-step을 예측하는 것)으로 TKGC와는 다른 방법론이다. 
    * 

## References
Jin, W., Qu, M., Jin, X., & Ren, X. (2019). Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. arXiv preprint arXiv:1904.05530.
Han, Z., Ma, Y., Wang, Y., Günnemann, S., & Tresp, V. (2020). Graph hawkes neural network for forecasting on temporal knowledge graphs. arXiv preprint arXiv:2003.13432.
Sankar, A., Wu, Y., Gou, L., Zhang, W., & Yang, H. (2020, January). Dysat: Deep neural representation learning on dynamic graphs via self-attention networks. In Proceedings of the 13th International Conference on Web Search and Data Mining (pp. 519-527).
