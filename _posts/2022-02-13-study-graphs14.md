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

## Abstract
Knowledge graphs enable a wide variety of applications, including question answering and information retrieval. Despite the great effort invested in their creation and maintenance, even the largest (e.g., Yago, DBPedia or Wikidata) remain incomplete. We introduce **Relational Graph Convolutional Networks (R-GCNs)** and apply them to two standard knowledge base completion tasks: Link prediction (recovery of missing facts, i.e. subject-predicate-object triples) and entity classification (recovery of missing entity attributes). R-GCNs are related to a recent class of **neural networks operating on graphs**, and are developed specifically to deal with the **highly multi-relational data characteristic of realistic knowledge bases**. We demonstrate the effectiveness of R-GCNs as a stand-alone model for entity classification. We further show that factorization models for link prediction such as DistMult can be significantly improved by enriching them with an encoder model to accumulate evidence over multiple inference steps in the relational graph, demonstrating a large improvement of 29.8% on FB15k-237 over a decoder-only baseline.

## 1. Introduction 
* statistical relational learning (SRL)에서 knowledge bases의 missing information을 에측하는 것이 가장 중요함. 
* Knowledge bases는 directed labeled mutigraph로, entity가 node가 되며, triples는 labeled edges로 encoded된 것이다. 
  ![image](https://user-images.githubusercontent.com/60350933/153741077-fe34a0fb-af73-44f8-adfc-2fb001e30dd6.png)

* SRL의 task
  * link-prediction (recovery of missing triples) 
  * entity classification (assigning types or categorical properties to entities)
* neighborhood 구조에 따라 graph에 missing information이 reside함. 
* **따라서, relational graph의 entity를 위한 encoder model을 개발** 
  * Classification model 
    * classification에 적용하기 위해서 graph의 각 node에 softmax classifier을 사용
    * Relational graph convolution network (R-GCN)을 사용하여 node representation을 추출하고 label을 predict하며, R-GCN의 파라미터는 cross-entropy loss를 optimize함으로 학습이 된다. 
  * Link prediction
    * autoencoder로 encoder와 decoder로 구성되어 있음
      * encoder: R-GCN을 통해 entity의 latent feature representation이 producing 됨
      * decoder: tensor factorization model (어떤 type의 factorization인지는 관게없지만 해당 모델에서는 DistMult 사용) - exploiting the representations to predict the edges 

* Contribution
  * GCN framework가 link prediction이나 entity classification task와 같은 relational data를 modeling하는데 적용될 수 있음을 처음으로 보임
  * parameter sharing을 위한 기술과 sparsity constraints를 강화하기 위한 기술들을 보일 수 있으며, 이를 기반으로 관계가 많은 multigraph에 R-GCN을 적용해볼 수 있다. 
  *  DistMult와 같은 factorization model의 성능이 향상됨

## 2. Neural relational modeling
#### 2.1. Relational graph convolutional networks
* local graph neighborhoods에 적용가능한 GCN을 large-scale relational data로 확장한 것
* 확장된 형태는 미분가능한 message-passing framework의 특수 케이스이다. 
* message-passing framework
  ![image](https://user-images.githubusercontent.com/60350933/153755832-d30d6289-bb8d-4c2c-9c04-40e946d78608.png)
  * gm(.,.)은 (message-specific) neural network와 유사한 함수이거나, linear transformation with weight matrix이다. 
    * local, structured neighborhood의 feature을 encoding하는데 유리하다. 
*  이를 기반으로, relational (directed and labeled) multi-graph의 entity의 forward-pass update를 계산하는 simple propagation model 정의
    ![image](https://user-images.githubusercontent.com/60350933/153756032-caace330-6490-46c5-ac52-ffb3e07dbf5f.png)
   * 이 때, normalized sum을 통해 neighboring nodes의 transformed feature vector를 accumulate함
  *  *(차별점)*  (edge의 type과 방향에 따른) relation-specific transformation을 도입하고, layer l+1에서의 node가 layer l에서의 representation의 information을 얻기 위해 data의 각 node의 특정 relation type에 single self-connection을 추가한다. 

* neural network layer update는 식 (2)을 각 node별로 parallel하기 evaluating하는 내용으로 구성되어 있다. 이는 neighborhood간의 explicit summatoin을 피하기 위한 용도로 sparse matrix multiplication에 효율적으로 사용할 수 있다. 여러 relational steps에 걸쳐 dependencies를 허용하기 위해 multiple layer이 쌓여질 수 있다. 이러한 graph encoder model을 R-GCN으로 칭함
* Simple node update에 대한 computation graph
  ![image](https://user-images.githubusercontent.com/60350933/153758520-42250d63-cb79-45c5-974e-007bdf594864.png)

#### 2.2. Regularization
* 식 (2)를 multi-relation data에 적용하는 것에 대한 핵심 문제는 graph의 relation 수에 따라 parameter의 숫자가 급격하게 증가하는 것
  * 희귀한 relation이나 매우 큰 모델에 대해 overfitting이 발생할 수 있음

* 해결방안: R-GCN layer의 weight을 regularizing하는 2개의 방법
  * 1) basis-decomposition
    ![image](https://user-images.githubusercontent.com/60350933/153758655-c5f3ffca-f6ad-4949-ac3b-53303ef76a67.png)
    * coefficients a가 r에만 의존하는 basis transformation ![image](https://user-images.githubusercontent.com/60350933/153758698-fd2cd854-58f1-429b-9dfc-ac35ae21c270.png) ![image](https://user-images.githubusercontent.com/60350933/153758708-f10239e8-25e4-4a89-85cb-a035900e5e2b.png)의 linear combination과 유사함
    * 다른 relation type 간의 effective weight sharing의 형태로 볼 수 있음
  * 2) block-diagonal-decomposition 
    ![image](https://user-images.githubusercontent.com/60350933/153758791-34ba4870-43b5-4ed2-8b8b-54c1f2387751.png)
    * direct sum over a set of low-dimensional matrices, which makes W the block-diagonal matrices
      ![image](https://user-images.githubusercontent.com/60350933/153758829-1a8563f1-a02c-4afc-9734-10895967dfb2.png)
    * 각 relation type의 weight matrices에 대한 sparsity constraint로 볼 수 있음
    * latent features는 within group 끼리 더 tightly coupled 되로고 설계
  * 이 2개의 decomposition은 highly multi-relational data에 필요한 파라미터의 개수를 줄여주며, basis- parameterization은 희귀한 관계와 빈도가 높은 관계 간의 parameter update를 가능하게 하기 때문에 overfitting이 줄어들 수 있다. 

* **R-GCN 모델의 전체 구조**
  * 1) 식(2)에서 정의된 것 (이전 레이어의 output이 다음 레이어의 input)과 같이 L개의 layer을 쌓아준다. 
    * 이 때 첫번째 layer의 input은 그래프의 각 node에 대한 unique one-hot vector이 될 수 있다. 각 node에 대한 bag-of-words 가 존재하는 경우에 pre-defined feature vector를 사용할 수 있다. 
  * 2) block representation을 위해서 one-hot vector를 single linear transformation을 위해 dense representation으로 mapping해준다. 

![image](https://user-images.githubusercontent.com/60350933/153760319-0d18e8c5-0547-4d38-8f36-ca74045bf095.png)

#### 2.3. Entity classification
* R-GCN layer을 쌓고 last layer의 output에 각 node 별 softmax를 취하여 classification을 진행하며, 모든 labeled nodes에 대해서 cross-entropy loss를 계산한다. 
![image](https://user-images.githubusercontent.com/60350933/153759260-42e8785f-c098-4ffb-9e5f-a537e7559584.png)

#### 2.4. link prediction
* Prediction of new facts
* **Graph auto-encoder model** - entity encoder와 scoring function(decoder)로 구성 제안
  * Encoder: 각 entity를 real-valued vector ei에 mapping
    * 기존연구: single, real-value ei를 각 node마다 사용하여 train시킴
    * 제안: R-GCn encoder을 통해 representation을 계산
      ![image](https://user-images.githubusercontent.com/60350933/153760477-3c41d274-d8fb-4dad-b3bc-3f1a3736e08b.png)
    *  
  * Decoder: edges를 vertext representation에 construct하며, (subject,relation, object)-triples를 function *s* 을 통해 scoring
    ![image](https://user-images.githubusercontent.com/60350933/153760423-e9dc681e-8a89-4366-9196-d95f7bd19c72.png)
    * DistMult를 score function으로 사용

* DistMult
  * 각 관계 r은 diagonal matrix R과 연관되어 있음
  * triple (s,r,o)의 score를 계산
    ![image](https://user-images.githubusercontent.com/60350933/153760529-3628875d-03d5-42c4-bdc8-0bab010a95ad.png)
  * negative sampling을 통해 model 학습
    * positive example의 subject나 object를 랜덤하게 훼손시켜서 sample

## 7. Conclusion
* Contribution
  * link prediction에서 DistMult factorization을 적용한 R-GCN이 그냥 factorization model을 optimize하는 것보다 더 좋은 성능을 가질 수 있다. 실제로, 29.8%의 성능 향상이 FB15K-237 dataset에서 있었다. 
  * entity classification에서 R-GCN은 competitive, end-to-end trainable graph-based encoder의 역할을 할 수 있음
* Extension
  * Graph autoencoder model은 다른 factorization model (ex. ComplEx)와 함께 사용되어져야 하므로, 이러한 다른 combination으로 확장이 가능하다
  * R-GCN에 entity features를 바로 integrate하는 것도 가능하다
  * Subsampling techniques를 다양하게 시도하는 것
  * replace current form of summation over neighboring nodes and relation types with data-dependent attention mechanism
  * KG가 아닌, factorization model이 효과적인 다양한 응용분야로 확장이 가능함.  

