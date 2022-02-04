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
    * 이때, Ã = A+I로, A 인접행렬에 self-connetion이 더해진 것이며, D~의 경우 Ã의 degree matrix이다. in-degree diagonal matrix인 D의 역행렬을 통해서 A를 정규화할 수 있다. 
      ![image](https://user-images.githubusercontent.com/60350933/152503821-08dead36-a5bc-4dcd-a9a2-e6ee2e54a1c5.png)
    * W(l)은 학습가능한 weight matrix
    * 이 때, 주어진 그래프 구조에 따라 convolution operation이 이루어짐을 알 수 있고, node-wise 선형 transform인 H(l)W(l)을 제외하고는 학습이 불가능하다. 따라서, convolution layer은 node-wise linear transformation에 activation function σ를 취한 fixed convolution으로 구성된다. 
    * 본 연구에서의 framework에서는 다양한 인접행렬로부터 추출한 convolution을 통해, 다양한 convolution이 가능하다. 

#### 2) Learning meta-graphs
![image](https://user-images.githubusercontent.com/60350933/152506643-10913c20-ca44-4c8d-94a7-51e48a8a3c1e.png)

* 핵심 아이디어
  * meta-path 그래프를 학습할 때, 유용한 meta-path *P*가 특정한 순서의 edge type (t1,t2,...,t*l*)과 연결된 새로운 adjency matrix A_*P*를 edge type별로 인접행렬의 곱을 통해서 얻는다. 
    ![image](https://user-images.githubusercontent.com/60350933/152504829-1f0c1aee-f235-4a23-8923-815451d1beae.png)
  * GTN의 k번째 Graph transformer layer은 softmax 함수에서의 weight으로 1x1 convolution의 인접행렬(edge type)을 선택할 수 있게 된다. 
    ![image](https://user-images.githubusercontent.com/60350933/152505172-d4877b1a-341d-4f5b-8e2b-2056049dfd3e.png)
    * ![image](https://user-images.githubusercontent.com/60350933/152505331-ceb0a735-dd7c-43e2-87e0-988e3a664daa.png) 는 1x1 convolution의 parameter
    * ![image](https://user-images.githubusercontent.com/60350933/152505408-b9bae3a6-17e2-4de9-bb1a-9065cbbefba9.png) 는 인접행렬의 convex combination
   * Meta-path 인접행렬은 output과 이전 (k-1)번째의 GT layer의 output 행렬의 곱으로 계산되어진다. 
     ![image](https://user-images.githubusercontent.com/60350933/152505804-c3940966-b47f-46a1-ba30-d780fb517f27.png)
   * numerical stability를 위해 matrix는 normalized된다. 
     ![image](https://user-images.githubusercontent.com/60350933/152505910-fd76a469-04aa-4930-a58e-5f93ade68579.png)
     * 이 때 D는 이전 단계의 2개의 행렬의 곱의 output의 degree matrix이다. 
* GTN이 임의의 meta-path를 edge type과 path-length 기반으로 학습할 수 있는지 확인
  * 임의의 길이 k+1를 가지는 meta-path의 인접행렬 계산식
     ![image](https://user-images.githubusercontent.com/60350933/152506246-9ee2d888-ca8c-4e44-b8fd-c9170453bcb9.png)
     * ![image](https://user-images.githubusercontent.com/60350933/152506309-24d8e010-cef9-4de7-88ea-1fa3eb5b7fb4.png) : set of edge types
     * ![image](https://user-images.githubusercontent.com/60350933/152506369-62f8cf2b-2284-485d-8497-22f0b391de6c.png) : k번째 GT layer의 edge type tk에 대한 weight
  * α가 one-hot vector일 때, Ap는 모든 (k+1) meta-path 인접 행렬들의 weighted sum임
  ![image](https://user-images.githubusercontent.com/60350933/152506937-4c46592d-70b1-45a0-8971-aa95d45364c5.png)

* Issue
  * GT layer을 추가한다는 것은 meta-path의 길이를 늘리는 것이기 때문에, original edge를 고려하지 않게 된다.
    * 길이가 짧더라도 중요한 path일 수 있기 때문에, identity matrix I (A0=I)를 포함한다. 이를 통해 GTN은 최대 k+1의 길이인 모든 길이의 meta-path를 학습할 수 있게 된다. 

#### 3) Graph Transformer Networks 
* 다양한 multi-path를 동시에 고려하기 위해
  * Output channels를 1x1 filter로 설정하면 k번째 GT layer인 A(k)가 output tensor ![image](https://user-images.githubusercontent.com/60350933/152508349-d457482d-313c-48de-b250-4a22c006ef7a.png) 가 되고, weight vector ![image](https://user-images.githubusercontent.com/60350933/152508391-d556c064-4599-48b0-b3de-6cbc9148450e.png) 가 weight matrix Φ(k)가 된다. 
  * Eq(5)는 tensor equation으로 나타날 수도 있다. 
    ![image](https://user-images.githubusercontent.com/60350933/152508523-aadeee5d-0c9a-4f71-b3ae-9a0546d31dd8.png)
    * 이때, ![image](https://user-images.githubusercontent.com/60350933/152508628-6cd2c28b-93b4-4b25-9f2b-fc13f1522012.png) 이며 ![image](https://user-images.githubusercontent.com/60350933/152509093-c48382d6-a1ad-4af3-a31b-ced59b8194db.png) 는 두 tensor의 합의 output의 degree tensor임. 
  * K의 GT layer을 쌓은 후, multi-layer GNN이 각 채널의 output tensor에 적용이 되고, node representation Z를 업데이트 한다. 
    ![image](https://user-images.githubusercontent.com/60350933/152509707-44242f9f-7701-4f10-8e0b-bc892ed5f587.png)
    * || 는 concatenation operator, C=number of channels, Z(l)은 l번째 GNN layer에서의 node representation을 의미한다.
    * ![image](https://user-images.githubusercontent.com/60350933/152509890-78bd53b9-a036-4e33-968d-2eb3b1209701.png) 에서 A(k)의 c번째 channel에서의 self-loop을 가지는 인접행렬이다. 
    * W(l)은 channel 모두 공유되는 학슴가능한 weight matrix이며 Z(0)은 X의 feature matrix, f_agg는 channel aggregation 함수이다. 
  * Final node representation Z(l)은 downstream task에 사용될 수 있다. 
    * 본 연구에서 softmax layer에 dense layer를 적용하여 node representation에 사용한다. Node의 ground truth label을 사용하여 back propagation과 gradient desent의 cross-entropy를 최소화하도록 weight을 최적화한다. 

#### 4) Fast Graph Transformer Networks
GTN의 scalability issue를 해결하기 위한 것. 
* GTN의 scalability issue
  * GTNs는 meta-path의 새로운 인접행렬을 2개의 인접행렬의 곱으로 **explicit**하게 계산하고, 각 layer 별로 새로운 인접행렬을 저장한다. 
  * 따라서 huge computational costs와 large memory가 필요하며, GTN을 large graph는 적용하기에 어렵게 만든다. 
  * 이러한 문제를 해결하기 위해서 GTN의 enhanced version인 FastGTN을 통해 meta-path의 새로운 인접행렬을 저장하지 않는 implicit한 그래프 structure transform을 진행한다. 

![image](https://user-images.githubusercontent.com/60350933/152511781-e59d939e-6566-485a-bc38-e56b479b944e.png)

* Derivation
  * C=1로 가정
  * (*문제*) 
  * 하나의 GCN 레이어는 새로운 그래프 구조의 top에 얹어지고, channel aggregation function은 identity function과 동일하다. 이 때, GTN의 node representation Z는 다음과 같다. 
     ![image](https://user-images.githubusercontent.com/60350933/152512019-422ded3f-d396-4c78-b3ff-cc803cc944f6.png)
     * A(k)는 K개의 GT 레이어를 가지는 GTN의 새로운 인접행렬이고, X는 input feature이며, W는 GCN layer의 선형 변환, D의 역행렬은 (A(k)+I)의 degree matrix의 역행렬이다. 
     * GT layer에서 선택된 2개의 인접행렬을 곱한 후에 output 인접행렬을 정규화하는 것을 알 수 있다. 
   * Z는 다음과 같이 재구성이 가능하다. 
     ![image](https://user-images.githubusercontent.com/60350933/152512241-b5a7c12a-aba1-4464-a308-0e42412ffdc9.png)
     * huge adjacency matrix의 곱을 통해 computational mbottleneck이 사용됨을 알 수 있다.  
       ![image](https://user-images.githubusercontent.com/60350933/152512899-43f57014-da5c-4aa2-b646-95b8da491930.png)
   * (*해결*)
   * Matrix mutiplication의 associative property를 다음과 나타낼 수 있고, 이는 각 레이어에서 인접행렬의 행렬 곱 없이 differently constructed 인접행렬을 사용하여 feature transformation의 sequence를 통해 identical feature을 구할 수 있음을 의미한다. 
     ![image](https://user-images.githubusercontent.com/60350933/152512675-ad2a5a6e-e1bc-447e-8537-a0d5db2dce1e.png)
     * 하지만 2개의 인접행렬의 곱을 구할 수 없기 때문에 degree matrix ![image](https://user-images.githubusercontent.com/60350933/152514391-879c0ec3-94a0-4438-b31c-453ec674fc29.png) 를 구할 수 없다. 
   * Given condition of input data와 normalized matrices의 proposition을 통해 Eq. 11의 degree matrices를 identity matrices로 만들어 Eq. 11을 구할 수 있다. 
   * **Proposition 1**
     ![image](https://user-images.githubusercontent.com/60350933/152514659-1eece305-d0ee-46ae-8265-28989dfbde71.png)
     ![image](https://user-images.githubusercontent.com/60350933/152520018-42d380b6-ed5f-4b78-adf4-8fd80c70a0a4.png)
     * (ii)로 인해, 각 k-th 레이어에서 ![image](https://user-images.githubusercontent.com/60350933/152520306-4ff1f209-cefb-4e7d-b3bf-a26f28251a48.png) 가 identity matrix I가 되므로, 이에 따라 Eq 11.은 아래와 같이 기술 가능함
       ![image](https://user-images.githubusercontent.com/60350933/152522235-b22874cb-e03d-4d9c-82d1-ad60c40059bb.png)
     * (iii)을 통해서 Z는 아래와 같이 표현이 가능함
       ![image](https://user-images.githubusercontent.com/60350933/152522488-d14ccf3a-a331-4df0-97c9-2e0a05847f84.png)

   * 각 레이어가 하나의 인접행렬의 convex combination으로 구성되어있기 때문에, K-layers는 K개의 인접행렬을 만들어낸다. 
     ![image](https://user-images.githubusercontent.com/60350933/152522633-8977ca01-2539-4c37-83ff-f5da123ffc17.png)
   * **이는 수학적으로 FastGTNs는 GTN과 수학적으로 동일함을 의미한다**
   * 1부터 K까지의 순서를 뒤집고, 1/2를 hyper-parameter γ로 replace하면, FastGTN의 output은 다음과 같이 표현된다. 
     ![image](https://user-images.githubusercontent.com/60350933/152522855-c7da36a7-5c07-4d06-91e9-d67523fb220b.png)

 * Multi-Channel setting에서의 Fast GTN
   ![image](https://user-images.githubusercontent.com/60350933/152523883-a96cbdc4-46c0-4b78-8a7c-9424cf3dd369.png)
    * C(l)은 number of channels, Z(l)은 l번째 FastGTN 레이어의 node-representation, ![image](https://user-images.githubusercontent.com/60350933/152523947-234260fd-b985-4ef5-b3e2-638e9a283a3a.png) 은 l번째 FastGTN 레이어의 c번째 channel의 선형변환, A는 normalized 인접행렬의 set, α(l;k;c)는  l번째 레이어의 k번째 FastGT 레이어의 c번째 채널의 convolution filter을 의미하고, Z(0)은 feature matrix이다. 

#### 5) Non-local operations
GTN의 단점은 transformation이 existing relation의 구성에 한정지어 졌다는 것이다. KGT layer은 (K+1)-hop relation에 대해서만 edge를 만들 수 있고, node의 semantic proximity between nodes를 기반으로 remote relation을 생성해낼 수 없다. 따라서 **meta-path를 벗어난 node의 semantic proximity를 활용한 node feature를 포함할 수 있는 non-local operation**을 제안함
* Fast GCN에만 적용
* 각 l번째 FastGTN 의 layer에 k번째 FastGT layer에만 non-local 인접행렬 ![image](https://user-images.githubusercontent.com/60350933/152524692-3d80d063-773c-4c19-bbb4-ef45b68dbc3c.png) 을 이전 FastGT 레이어의 hidden representation Z(l,k-l)을 기반으로 구성하고, 이 non-local 인접행렬을 인접행렬 후보 set A에 붙여, non-local relation을 활용할 수 있도록 한다.
* non-local 인접행렬 구축
  ![image](https://user-images.githubusercontent.com/60350933/152525248-87c1faf3-6550-4ae5-91eb-3d3b9dd61a6c.png)
  * Graph affinity matrix를 node similarity기준으로 l번째 FastGT layer마다 계산
  * k-1번째 FastGT layer의 multi=channel hidden representation의 유사도를 구하고, 이를 latent space에 non-linear transformation gθ 를 통해 project한다. 
  * 이후 latent의 similarity를 통해 affinity matrix를 계산한다. 
    * GAE를 통해 similarity 계산
  * Affinity matrix는 fully connected graph의 weighted 인접행렬로 볼 수 있고, 해당 matrix 계산은 computing cost가 크기 때문에 각 node i  별로 n개의 가장 큰 weight을 추출하여 affinity matrix를 sparsify한다. 
    ![image](https://user-images.githubusercontent.com/60350933/152525581-584a1168-996b-431b-9a99-b7a8bfa282f8.png)
  * 각 행의 edge weight에 softmax function을 적용하여 non-local adjacency matrix를 row-wise로 normalize한다. 
    ![image](https://user-images.githubusercontent.com/60350933/152525784-6617a3a1-4de1-4991-8d7d-b526fd76aa93.png)
  * 이를 transformation에 사용하기 위해서 non-local parameter을 각 FastGT layer의 1x1 convolution filter에 사용한 후, k번째 FastGT layer의 인접행렬 set에 append한다. 

#### 6) 다른 GNN 아키텍처와의 비교
* 1. GCN과의 비교 시 아래 두 가지 사항은 동일하다. 
  * GCN의 l번째 GCN layer의 output node representation
    ![image](https://user-images.githubusercontent.com/60350933/152526302-80fed290-671b-40a3-b251-86986bcc7a58.png)
  * FastGTN의 FastGT layer의 개수가 1이고, channel의 개수가 1인 경우, node representation, 
    ![image](https://user-images.githubusercontent.com/60350933/152526288-96ac6b3a-b216-4ba2-bbe8-477e3eb5fd7a.png)
* 2. Mixhop은 FastGTN의 special case
* 3. RGCN 또한 FastGT 레이어가 하나이고, channel의 개수가 basis matrices와 동일한 경우, different linear combination을 적용한다는 것을 빼면 동일하다. 

## 4. Experiments
* Q1. How effective are the GTNs and FastGTNs with non-local operations compared to state-of-the-art GNNs on both homogeneous and heterogeneous graphs in node classification?
* Q2. Can the FastGTNs efficiently perform the identical graph transformation compared to the GTNs?
* Q3. Can GTNs adaptively produce a variable length of meta-paths depending on datasets?
* Q4. How can we interpret the importance of each metapath from the adjacency matrix generated by GTNs?

#### 1) Experimental Setting
* Dataset
  ![image](https://user-images.githubusercontent.com/60350933/152527218-24d0f689-627e-46ae-88d9-4f210fca12ee.png)
* Baselines
  * MLP, Node2Vec, GCN, GAT, JK-Net, Mixhop, GCNII, RGCN, HAN, HGT
* Implementation with PyTorch
  * Dimentionality of hidden representations: 64
  * Adam optimizer
  * hyper-parameter search
    * learning rate from 1e-3 to 1e-6
    * dropout rate: 0.1 to 0.8
    * epoch: 50 to 200. 
  * micro-F1 scores are computed
#### 2) Results
![image](https://user-images.githubusercontent.com/60350933/152527641-82f974a0-4cf6-4b8b-aacf-a8eae631fc0f.png)
![image](https://user-images.githubusercontent.com/60350933/152527671-5dc1cc1b-2ce2-4ffe-b1bc-24776ab87821.png)
* 모든 데이터에서 GTN과 FastGTN이 best performance를 보임


#### 3) Efficiency of FastGTN
![image](https://user-images.githubusercontent.com/60350933/152528050-30c46852-a3c7-4975-8eaa-83444d805737.png)
* Predictions (confidence scores) by the FastGTN과 GTN이 동일함
![image](https://user-images.githubusercontent.com/60350933/152528150-6bc96dd4-cefc-43c3-a5d2-f0538859e5ab.png)
* FastGTN이 GTN보다 computational cost와 memory 사용량에 있어서 효율적임. 특히 graph가 더 커질 수록 더 performance gain은 큼

#### 4) Interpretation of GTN
![image](https://user-images.githubusercontent.com/60350933/152528358-7eac5a4c-a6bb-49a9-827e-e3f3761ae379.png)
* GTN에서도 table 5의 사전 정의된 meta-paths를 잘 예측함. 
* GTN에서는 사전 정의되지 않은 중요한 meta-path 또한 잘 예측함. 
  * DBLP에서 CPCPA를 가장 중요한 meta-path라고 정의하는데 이는 사전 정의가 되지 않는다. 
  * Author의 research 분야가 venue와 연관되어있음을 의미하는데 이는 domain 적으로 의미가 있다. 
* GTN은 short meta-path를 학습하는데 더 유용하다. 특히 high-attention scores를 identity matrix에 assign하면 할 수록 deeper layer에서 더 길이가 짧은 meta-path를 중요하게 여기는 경향이 있음. 

## 5. Conclusions
* Future directions
  * Applying GTNs to other tasks such as graph classification and link prediction. 
 



 


