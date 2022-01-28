---
layout: post
title:  "[논문리뷰] Graph Partition Neural Networks For Semi-Supervised Classifications "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Liao, R., Brockschmidt, M., Tarlow, D., Gaunt, A. L., Urtasun, R., & Zemel, R. (2018). Graph partition neural networks for semi-supervised classification. arXiv preprint arXiv:1803.06272.

## Abstract
We present **graph partition neural networks (GPNN)**, an extension of graph neural networks (GNNs) able to **handle extremely large graphs**. GPNNs alternate between **locally propagating information between nodes in small subgraphs and globally propagating information between the subgraphs**. To efficiently partition graphs, we experiment with several partitioning algorithms and also propose a novel variant for fast processing of large scale graphs. We extensively test our model on a variety of semi-supervised node classification tasks. Experimental results indicate that GPNNs are either superior or comparable to state-of-the-art methods on a wide variety of datasets for graph-based semi-supervised classification. We also show that GPNNs can achieve similar performance as standard GNNs with fewer propagation steps.

## 1. Introduction
* 목적: graph-structured의 input을 high-capacity neural network-like models로 학습시키는 것
  * Graph Neural Network가 가장 기반이 되는 모델이고, 현재 graph data에 neural network와 유사한 모델들을 적용하는 시도 존재
* GNN 모델의 한계
  * 정보가 graph에 어떻게 propagated되는지에 대해서 설명하지 못한다. 
    * sequence의 element간의 추가적인 relationships으로 augmented된 sequence가 있을 때, graph 상에서 long distances에서 정보가 어떻게 propgated되어야 할지에 대한 시나리오만 존재할 뿐이다. 
  * **Synchronous message-passing systems**을 GNN이 대부분 따르는데, 이 방법론은 적용하기는 쉽지만, graph 상에서 long distances에 걸쳐 정보가 있는 경우에 inefficient하다
    * N길이의 sequence에 대한 schedule을 짤 때, 총 O(N^2)의 messages가 information propagation에 필요하며, 이 모든 message는 memory에 저장이 되어야 한다. 
    * 일반적으로 sequence data를 다룰 때 O(N)이 소요되는 **forward pass-backward pass**를 사용하여 end-to-end로 information을 propagate하게 된다 (ex. bi-directional RNN) 
    
  * propagate information over the graph following some pre-specified sequential order (as in Bidirectional LSTMs)
    * (*problems1*) if graph used for training has large diameter, the unrolled GNN computational graph will be large, and will lead to vanishing/exploding gradients and resource contraints
    * (*solution1*) input graph의 모든 node와 연결이 된 'dummy node'를 사용하지만, additional node와 edge로 인해 포함되었던 정보의 왜곡이 발생함
    * (*problems2*) sequential schedules are typically less amenable to efficient acceleration on parallel hardware. 
    
* 제안
  * Graph partitional neural network is proposed to exploit a propagation schedule combining features of synchronous and sequential propagation schedules
    * Partition the graph into disjunct subgraphs and a cut set
    * Synchronous propagation within subgraphs with synchronous propagation within the cutset

## 2. Model
* focuses on the directed graphs
  * undirected graphs can be applied to by splitting undirected edge into two directed edges
  * Pre-specified edge types: node간 다양한 관계를 encode하기 위해서 사용됨. 
### 2.1. Graph Neural Networks
* Extension of RNN to arbitrary graphs
* Each node *v* is associated with an initial state vector *h_v(0)* at the time step 0.  
* At time step t, an outgoing message is computed for each edge by transforming the source state according to the edge type
* At the receiving nodes, all messages are aggregated through either summation, average or max-pooling
* Every node will update its state vector based on its current state vector and the aggregated message
  * updated function may be a gated recurrent unit (GRU), LSTM, or a fully connected network
  * all nodes share the same instance of update function
* The described propagation step is repeatedly applied for a fixed number of time steps T, to obtain final state vectors
* A node classification task can be implmented by feeding these state vectors to a fully connected neural network which is shared by all ondes
  * Back-propagation through type (BPTT) is typically adopted for the learning model
    
### 2.2. Graph Partition Neural Networks
* GNN에서는, Observe synchronous schedule in which all nodes receive and send messages at the same time.
* 하지만, 그래프의 모든 node들이 message를 동시에 보내는 것이 아니기 때문에 different propagation schedules를 고려해야 한다. 
#### 2.2.1. propagation model
![image](https://user-images.githubusercontent.com/60350933/151506946-ab7a7153-7754-48a3-b9d2-b0641f5566e2.png)
* Figure 1
  * Full synchronous propagation schedule requires significant computation at each step
  * Sequential propagation schedule results in sparse and deep computational graphs
    * require multiple propagation rounds across the whole graph, resulting in an even deeper computational graph
  
* New propagation method (효율적이고 tractable learning이 가능함)
  * partition the graph into disjunct K subgraphs that each contains a subset of nodes and the edges induced by this subset. 
  * Cut set- the set of edges that connect different subgraphs- is also prepared. 
  * Alternate between propagating information in parallel local to each subgraph and propagating messages between subgraphs
  ![image](https://user-images.githubusercontent.com/60350933/151508109-5652b0ae-69a0-4180-8bd4-76631b45d30e.png)
  
  * The number of messages
    * Synchronous propagation uses 5*10 messages (Fig. 1(d))
    * Partitioned propogation: 36 messages (Algorithm 1)  
  * 일반적으로, partitioned propagation에서는 synchronous schedule 보다 더 적은 message를 boroadcast하는데 사용하며, sequential schedule에서 요구되는 deep computational graphs를 사용하지 않는다. 

#### 2.2.2. Graph Partition
* Re-use the classical spectral partition method especially **normalized cut method** and use random walk normalized graph Laplacian matrix.
* 문제: spectral partition method는 large graph에서 느리고 scale 하기 어렵다. 
* 해결방안: Heuristic method based on multi-seed flood fill partition algorithm
![image](https://user-images.githubusercontent.com/60350933/151509380-bf979f91-d4ed-43d3-9186-5c926ae480d5.png)

  * 1. Randomly sample the initial seed nodes biased towards nodes which are labeled and have a large out-degree
  * 2. Global dictionary assigning nodes to subgraphs and initially assign each selected seed node to its own subgraph
  * 3. Grow the dictionary using flood fill, attaching unassigned nodes that are direct neighbors of a subgraph to that graph. 
  * 4. To avoid the bias towards the first subgraph, we randomly permute the order in the beggining of each round
  * The procedure is repeatedly applied until no subgraph grows anymore. 
  * The disconnected components left in the graph is assigned to the smallest subgraph found so far to balance subgraph sizes

#### 2.2.3. Node features & classification
* graph-structured data의 한계
  * 1) do not have observed features associated with every node
  * 2) have dimensional sparse features per node
* Initial node label을 위한 two type of models
  * **Embedding-input**
    * learnable node embedding을 사용하여 한계 1을 해결하고자 함 
    * For the nodes with observed features, we initialize the embeddings to these observations, and all other nodes are initialized randomly
    * All embeddings are fed to the propagation model and are treated as learnable parameters
  * **Feature-input** 
    * Sparse fully-connected network이 한계 2를 해결하기 위해서 사용됨
    * Dimension-reduced feature은 propagation model에 사용되고, sparse network은 rest of the model과 함께 학습이 된다.  

## 4. Experiments
#### 4.1. Experiments
* 사용한 dataset
  * citation network 기반 document classification
    
  * knowledge graph에서 추출된 bipartite graph의 entity gclassification
    * NELL dataset 
   
  * distantly supervised entity extraction에 적용함. 
    * DIEL dataset 
    ![image](https://user-images.githubusercontent.com/60350933/151510404-38792705-e8d4-45f8-b2dd-a0bf59c43994.png)

#### 4.5. Comparison of different propagation schedules
* sequential order과 series of minimum spanning trees (MST)를 기반으로한 schedules
* Sequential order
  *  Graph traversal via bread first search (BFS) which gives visiting order
  *  split the edges into those that follow the visiting order and those that violate it
  *  The edges in each class construct a direced acyclic graph (DAG), and we construct a propagation schedule from each DAG following that every node will send messages once it receives all the messages from its patents and updates its own state. 
  *  This sequential schedule reduces to a standard bidirectional RNN on the chain graph
* MST schedule
  * Find a sequence of minimum spanning trees
    * Assign random positive weights (0-1) to every edge
    * Apply Kruskal's algorithm to find MST
    * Increase the weights by 1 for edges which are present in the MST. 
    * Iterate the process until we find K MST (k=total number of propagation steps)
* Cora dataset으로 synchronous, partition-based propagation schedules, sequential order, MST schedule을 비교
  * 비교
    * Synchronous: every node sent and received messages once and updated its state
    * Sequential: messages from all roots of the two DAGs were sent to all the leaves
    * MST-based: sending message from the root to all leaves on one minimum spanning tree
    * Partition: one outer loop of the algorithm
  * Sequential schedule에서 one propagation step에서 message가 propagated furthest. 
  * When increasing the number of propagation steps, it performs worse as the deep computational graph makes the learning hard

## 5. Conclusion
* Relying on graph partitions, the model alternates between locally propagating information between nodes in small subgraphs and globally propagating information between the subgraphs. 
* Future research directions
  * Learn graph partitioning as well as the GNN weights using **soft partition assignment**
  * Probalistic graphic model에서 사용된 propagation schedules 또한 GNN 맥락에서 고려해볼 수 있다. 
  * graph내의 서로 다른 node들이 memory를 효율성 증진을 위해 공유할 수 있다. 



