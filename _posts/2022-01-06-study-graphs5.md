---
layout: post
title:  "[그래프데이터분석및응용] Random walk based network embedding"
subtitle:  "Random Walk"
categories: study
tags: graphs
comments: true
header-img:

---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. End-to-end Node Embedding
#### 1.Traditional machine learning for graphs
   * Given input graph
   * Extract node, link, and graph-level features
   * learn a model that maps features to labels 

#### 2. Graph representation learning
   * alleviates the need to do feature engineering every single time by **automatically learning the features**
   * Goal
     * Efficient **task-independent feature learning** for machine learning with graphs
     * 각 node를 잘 설명할 수 있는 vector을 찾는 것

#### 3. Why embedding?
   * **Task**: Map nodes into an embedding space
   * **Similarity of embeddings between nodes**=similarity in the network
      * ex: Both nodes are close to each other (connected by edge)
   * Encode the network information
   * Potentially used for **many downstream tasks*

#### 4. Node embeddings 
   * Setup 
     * Assume that we have graph G
     * *V* is the vertext set
     * *A* is the binary adjacency matrix
     * For simplicity, assume that there is no additional information 
   * Goal: encode nodes so that similarity in the embedding space approximates similarity in the graph
     ![image](https://user-images.githubusercontent.com/60350933/148395758-24158e06-c56c-459f-947a-3db60c422417.png)
      * embedding space의 vector의 dot product가 original network의 similarity. 
      * Original network에서의 similarity는 무엇인가? 
   
   * Overall Steps
     * 1. **Encoder** maps from nodes to embeddings
     * 2. Define a node **similarity function**
       - i.e. a measure of similarity in the original network
     * 3. Decoder DEC maps from embeddings to the similarity score in the original network
     * 4. Optimize the parameters of the encoder so that: 
       - ![image](https://user-images.githubusercontent.com/60350933/148396443-14e24f22-9a76-47ff-864e-e9f1dd1e0a63.png)
     * 이 때, two key components는 Encoder과 Similarity function이다.  
   
   * "Shallow" Encoding
     * Encoder is just and embedding-lookup
       *각 node에 대한 embedding이 되는 지를 사전 형식으로* 
     * matrix: each column is a node embedding 
     * vector: indicator where all zeroes except a one in column indicating node v. 
       ![image](https://user-images.githubusercontent.com/60350933/148397099-90a8d4a5-8b84-4ce3-bcaa-1f52e58e8180.png)
     * Each node is assigned a unizue embedding vector
     * Representative methods:
       - **1. DeepWalk**
       - **2. Node2Vec**
       
   * How to define the similarity in the original network? 
     * **Similarity function** specifies how the relationships in vector space map to the relationships in the original network. 
  
#### 5. Framework Summary
   * Encoder+Decoder Framework
   * Shallow encoder: embedding lookup
     * parameters to optimize: Z which contain node embeddings for all nodes
   * Deep encoder: Graph Neural networks
   * Decoder: based on node similarity
   * Objective: Maxmize dot product of node embeddings for node pairs (u,v) that are similar
   
## 2. Node similarity definition that uses random walks
#### 1.Overview
   * unsupervised/self-supervised way of learning node embeddings
   * Task independent
 
#### 2. Formulazing the optimizing problem
   * Three components to define
     ![image](https://user-images.githubusercontent.com/60350933/148398321-2a6ce80a-03b7-4c2a-a024-c81e6f6b74ab.png)
     * 1) Decoder function
     * 2) S[u,v]: A graph based similarity measure between nodes
     * 3) *l*: A loss function measuring the discrepancy between the decoded similarity values and the true values

#### 3. Random Walk
   * The (random) sequence of points visited by random selection
   * Random walk embeddings
     * probability that u and v co-occur on a random walk over the graph
     * 이 확률을 embedding된 space에서의 dot product와 동일하게 하는 것이 목표이다.
     * Steps
       * 1. Estimate probability of visiting node v on a random walk starting from node *u* using some random walk strategy R
       * 2. Optimize embeddings to encode these random walk statistics 
   * 왜 Random walk를 사용하는가?
     * Expressivity: Flexible stochastic definition of node similarity that incorporates both local and higher-order neighborhood information
       * Idea: if random walk starting from node u visits v with high probability, u and v are similar (high-order multi-hop information)
     * Efficiency: Do not need to consider all node pairs while training and need to consider only the pairs that co-occur on random walks 
       * 특히 network size가 클 때 유용함
   * Unsupervised feature learning
     * Idea: Learn node embeddings such that nearby nodes are close together in the network
     * Given a node u, 
       N_R(u): Neighborhood of u obtained by some random walk strategy R
   * Strategies used to run random walks
     * Deepwalk: run fixed-length, unbiased random walks from each node
#### 4. Optimization
   * P_R(v|u): probability of visiting v on a random walk starting with u with random walk strategy R
   * maximize 
     ![image](https://user-images.githubusercontent.com/60350933/148399894-ac9036d3-da62-42b9-91f3-81ae1c05c2a5.png)
   * Goal: learn embeddings so that the following holds 
     ![image](https://user-images.githubusercontent.com/60350933/148400016-f8184c19-f679-4484-aeb1-567e114bed85.png)
     
   * The probability is parameterized by softmax function
     ![image](https://user-images.githubusercontent.com/60350933/148400148-5818b1f3-b301-4009-b3ab-ed16c7e72c6d.png)
     - 0에서 1 사이의 값으로 표현하기 위하여 softmax function을 사용하는 것이다. 
   
   * Negative Sampling
     * However, the normalization term from the softmax is the culprit 
     * Instead of normalizing w.r.t all noes, just normalize against *k* random 'negative samples'. 
     ![image](https://user-images.githubusercontent.com/60350933/148403503-a6369c40-ec1b-4472-8337-9f586fbc8bb0.png)
     * sample *k* negative nodes each with probability proportional to its degree
     * Two considerations for k
       - Higher k gives more robust estimates
       - Higher k corresponds to higher bias on negative events
       - from 5 to 20
       
   * Stochastic Gradient descent를 사용하여 optimize를 할 수 있음
   
## 3. Node2Vec (other random walk method)
#### 1. Overview
   * Key observation: Flexible notion of network neighborhood of node u, N_R(u) leads to rich node embeddings
   * develop biased 2nd order random walk strategy R to generate network neighborhood N_R(u)
   * Idea: use flexible, biased random walks that can trade off between local and global views of the network
  
#### 2. Biased Walks
   * Two classic strategies
     * First choose direct neighbors (BFS) and then choose indirect neighbors (DFS)
       - BFS: micro-view of neighborhood
       - DFS: macro-view of neighborhood
       ![image](https://user-images.githubusercontent.com/60350933/148407603-54c4539d-c422-4b99-839f-d2c991f12f3c.png)

   * Interpolating BFS and DFS
     * Biased fixed-length random walk R that given a node u generates neighborhood N_R(u)
     * Two hyperparameters
       - **Return parameter** *p*: return back to the previous node
       - **In-out parameter** *q*: moving outwards (DFS) vs inwards (BFS)
         *q is the ratio of DFS and BFS*

#### 3. Node2Vec algorithm
   * 1. Compute random walk probabilities
   * 2 Simulate r random walks of length *l* starting from each node u
   * 3. optimize the node2vec objective using stochastic gradient descent
   * 특징
     * Linear time complexity 
     * All the 3 steps are individually parallelizable


## 4. Other random walk ideas
#### Different kinds of biased random walks
   * Metapath2vec
     * based on the node attributes 
     * based on learned weights
   * Alternative optimization schemes
     * LINE: directly optimize based on 1 and 2-hop random walk probabilities 
   * Network preprocessing technichs 
     * Struc2vec, HARP: run random walks on modified versions of the original network. 

## 5. Limitations of shallow embedding methods
#### 1. Limitations
   * O(|V|) parameters are needed
     * no sharing of parameters between nodes
     * Every node has its unique embedding
   * transductive
     * Cannot generate embeddings for nodes that are not seen during training
   * Do not incorporate node features
     * Many graphs have features that we can and should leverage. 
    
