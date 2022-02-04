---
layout: post
title:  "[논문리뷰] Graph Transformer Networks "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Yun, S., Jeong, M., Kim, R., Kang, J., & Kim, H. J. (2019). Graph transformer networks. Advances in Neural Information Processing Systems, 32, 11983-11993.

## Abstract
Graph neural networks (GNNs) have been widely used in representation learning on graphs and achieved state-of-the-art performance in tasks such as node classification and link prediction. However, most existing GNNs are designed to learn node representations on the fixed and homogeneous graphs. The limitations especially become problematic when **learning representations on a misspecified graph or a heterogeneous graph that consists of various types of nodes and edges**. In this paper, we propose **Graph Transformer Networks (GTNs)** that are capable of generating new graph structures, which involve identifying useful connections between unconnected nodes on the original graph, while learning effective node representation on the new graphs in an end-to-end fashion. **Graph Transformer layer**, a core layer of GTNs, learns a soft selection of edge types and composite relations for generating useful multi-hop connections so-called meta-paths. Our experiments show that GTNs learn new graph structures, based on data and tasks without domain knowledge, and yield powerful node representation via convolution on the new graphs. Without domain-specific graph preprocessing, GTNs achieved the best performance in all three benchmark node classification tasks against the state-of-the-art methods that require pre-defined meta-paths from domain knowledge.

## 1. Introduction
#### 1) 배경
* Graph Neural Network (GNN)은 graph 구조의 데이터의 representation을 학습하기 위한 중요한 도구가 됨
* 한계
  * GNN은 graph가 고정되어있고 homogeneous함을 가정함
  * 없거나 거짓의 연결이 존재하는 noisy graph는 graph에 잘못된 neighbor을 기반으로한 ineffective convolution으로 이어진다. 
  * 다양한 형태의 nodes나 relation을 기반으로 하는 heterogeneous graph에는 GNN을 적용하기가 어렵다. 
    * heterogeneous graph의 경우 node type이나 edge type의 중요도는 task에 따라서 달라지고 어떤 경우에는 아예 쓸모가 없을 수 있다
    * heterogeneous graph를 다룰 때 homogeneous graph처럼 하나의 node와 edge로만 구성되었다고 볼 수도 있지만 모든 information에 대한 정보를 다 처리하는 것이 아님
    * 따라서 이러한 그래프를 다루기 위해서 **meta-path**(paths connected with heterogeneous edge types)를 직접 디자인을 하기 시작하였고, heterogeneous 그래프를 meta-path로 구성된 homogeneous graph로 transform하였음
    * Meta-path를 수동으로(?) 그려주는 것은 비효율적이다

#### 2) Proposal 1 - Graph transformer networks (GTNs)
* 기존의 그래프에서 유용한 multi-hop connection (e.g. meta-path)을 포함하고 noisy connection을 배제하는 새로운 graph로 기존의 그래프를 transform하도록 학습
* 새로운 그래프의 node representation을 end-to-end로 학습
* **Graph Transformer Layer** 은 edge type의 인접행렬의 soft selection을 학습하고, 유용한 meta-path가 생성될 수 있게 2개의 선택된 인접행렬을 곱한다. 
  * *soft selection* : 
* Identity matrix를 leveraging함으로서, heterogeneous graph의 선택된 edge type과 연결된 arbitrary-length composite 관계들을 기반으로 한 새로운 그래프 구조를 생성할 수 있음

#### 3) GTN의 issue 해결방안
* FastGTN
  * 큰 크기의 adjacency matrix를 곱해야 하기 때문에 computation costs와 large memory를 고려한 모델 제안
  * FastGTN의 경우 2개의 인접행렬의 곱 없이 graph를 implicitly transform한다. 
  * 기존 GTN에 비해서 230배 빠르고 100배 적게 메모리를 사용하지만, GTN과 동일한 graph transformation이 가능케 하도록 한다. 
* Non-local operations
  * GTN의 edge generation이 input graph의 meta-path의 nodes 연결에만 의존하였는데, 이는 node의 semantic proximity를 고려하지 못한 것이다. 
  * non-local operation을 허용함으로서, node의 semantic proximity를 고려한 graph transformation을 가능하게 함. 

#### 4) Contribution
* GTN을 통해서 유용한 meta-path와 multi-hop connection을 판별하여 new graph structure을 학습할 수 있다. 
* Fast-GTNs를 통해서 인접행렬의 곱 없이 graph를 implicitly하게 transform이 가능하다
* meta-path외의 node 간의 semantic proximity를 활용할 수 있도록 non-local operations 활용
* 6개의 heterogenous와 homogeneous graph에서 node classification에서 SOTA 달성


## 2. Related Works
#### heterogeneous graph
* GNN with relation-specific parameters
  * R-GCN: employed GCN with relation-specific convolutions 
  * Heterogeneous Graph Transformer (HGT): parameterize the meta relation triplet of each edge type and transformer 아키텍처의 self-attention을 활용한 구조를 통해 다양한 관계의 구체적인 패턴을 학습함
* GNN with relation-based graph transformations (meta-path를 주로 활용하는 방법)
  * Heterogeneous graph attention network (HAN): 수동으로 선택한 meta-path를 통해 heterogeneous 그래프를 homogeneous graph로 transform시킨 뒤 그래프에 attention 기반 GNN 적용
    * multi-stage 접근법이고, 각 데이터 별로 meta-path를 지정해야 한다는 한계
    * 어떤 meta-path를 선택하느냐에 따라서 성능이 달라짐.   


## 3. Method
* 더 효율적인 graph convolution을 만들고 강력한 node representation을 학습하기 위해서 여러 후보 인접행렬을 찾음
* 새로운 그래프 구조를 학습한다는 것은 유용한 meta-path와 multi-hop connections를 인식하는 것을 포함함

#### 1) Preliminaries
* **Heterogeneous Graph**
  * Deach node v and each edge e가 각각의 type mapping function ( Tv(v): V → Tv, Te(e): ε→Te)과 연관되어 있는 Directed graph
    * G = (V, E, Tv, Te)
  * 해당 그래프는 인접행렬의 set이나 ![image](https://user-images.githubusercontent.com/60350933/152460708-58c70b04-4d2c-4874-b232-d64198332e43.png) tensor ![image](https://user-images.githubusercontent.com/60350933/152460838-ad067c46-0d07-41fc-a5c1-c050447796c4.png) 으로 나타날 수 있다. 
    *  이떄 At는 t 번째 edge type의 인접행렬이고, |V|=N
    *  At[i,j]는 node *j*로부터 node *i*까지의 t번째 edge type의 weight을 나타냄. 
       *  그래프가 single type의 node와 edge를 가지면,|Tv| = 1,  |Te| = 1인 homogeneour graph이다. 
 * **Metha-path**
   *  Heterogeneous 그래프에서의 multi-hop connection이며, heterogeneous edge type을 연결시켜주는 path
* **Graph Convolutional Network** 
  * 해당 모델에서 end-toend로 node를 classify할 때 사용이 됨.  
  * H(*l*)을 l번째 layer의 feature representation이라고 할 때, forward propagation은 다음과 같다. 
    ![image](https://user-images.githubusercontent.com/60350933/152461857-8bca7d7c-5d55-4884-849c-c2a5fdc8a262.png)
    * 이때, Ã = A+I로, A 인접행렬에 self-connetion이 더해진 것이며, D

