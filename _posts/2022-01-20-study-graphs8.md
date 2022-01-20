---
layout: post
title:  "[논문리뷰] Graph Attention Networks "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

## Abstract
We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

## 1. Introduction
* grid-like structure로 표현되지 않는 irregular domain에 있는 data가 많으며, 이런 데이터는 graph를 통해서 represented될 수 있다. 
* Generalizing convolutions to the graph domain
  * Spectral approach
    * spectral representation of graphs and applied in the context of node classification
    * the learned filters depend on the Laplacian eigenbasis, which depends on the graph structure
      * 특정한 그래프 구조에 맞게 학습된 모델은 다른 구조를 가진 graph에 바로 적용할 수 없다. 
  * Non-spectral approach
    * Define convolutions directly on the graph, operating on groups of spatially close neighbors
    * challenge: define an operator which works with different sized neighborhoods, maintains the weight sharing property of CNNs. 
    * 대표적인 모델: MoNeT, GraphSAGE
* Attention Algorithm
  * sequence-based task에서 사용됨
  * allow for dealing with variable sized inputs, focusing on the most relevant parts of the input to make decisions
  * **Self-attention**(intra-attention): when an attention mechanism is used to compute a representation of a single sequence.
* **Attention-based architecture to perform node classification of graph-structured data**
  * computation of hidden representations of each node in the graph, by attenting over neighbors (self-attention)
  * 특징
    * operation이 효율적임- it is parallelizable across node-neighbor pairs
    * it can be applied to graph nodes having different degrees by specifying arbitrary weights to the neighbors
    * directly applicable to inductive learning problem including tasks where the model has to generalize to completely unseen graphs
      
## 2. GAT Architecture
**Building block layer: used to construct arbitrary graph attention networks** 
#### 2.1. Graph attention layer
* input: set of node features 
  ![image](https://user-images.githubusercontent.com/60350933/150321438-a671de9b-4b14-4eeb-8c52-6ebe7feeb185.png)
  N = number of nodes
  F = number of features in each node
* output: new set of node features (of potentially different cardinality F')
* 1) shared (learnable) linear transformation
  * To optain sufficient expressive power to transform the input features into higher-level features
  * as an initial step, linear transformation, parameterized by weight matrix, W ∈ R_F'xF is applid to every node
* 2) After the transformation, self-attention is performed on the nodes - a shared attentional mechanism computes *attention coefficients* that indicate the importance of node *ㅓ
  ![image](https://user-images.githubusercontent.com/60350933/150323874-3f918ee2-5825-4c62-8fec-eaf89f4e5327.png)
* 3) The model allows every node to attend on every other node, dropping all structural information
* 4) masked attention: injecting graph structure into the mechanism
  * compute only e_ij for nodes j in N_i, where N_i is some neighborhood of node i in the graph
  * first-order neighbors of i will exist
* Coefficients computed by attention mechanism
  * Normalize all the nodes across all choices of j using softmax function to make coefficients easily comparable
  * Attention mechanism a is a single-layer feedforward neural network parametrized by a weight vector
  * LeakyReLU nonlinearity 
  ![image](https://user-images.githubusercontent.com/60350933/150365677-65a37a57-8c99-4af6-9db4-633e4187e7f2.png)
  * Used to compute linear combination of the features, which serve as final output feature for every node. 
  Equation 4. 
  ![image](https://user-images.githubusercontent.com/60350933/150367291-98a41913-918b-4f52-a6ff-bc0041780dff.png)
* **Multi-head attention**
  * to stability the learning process of self-attention
  * K independent mechanisms execute the transformation of Equation of 4 and then their features are concatenated
  * output feature representation
    ![image](https://user-images.githubusercontent.com/60350933/150368897-88c09ee3-3cec-4740-99e7-d7458c45db56.png)
    * Final returned output (h'): KF' feature로 이루어짐
  * Final layer에서는 concatenation이 sensible하지 않게 되기 때문에 averaging을 사용하며, final nonlinearity를 softmax나 logistic을 통해서 delay시킨다. 
    ![image](https://user-images.githubusercontent.com/60350933/150369671-b8f0493c-23ad-4810-bc16-a282acbedfd2.png)
 
#### 2.2 Comparisons related network
**장점** 
* 1) Highly efficient
  * Operation of the self-attentional layer can be parallelized across all edges 
  * computation of output features can be parallelized across all nodes
  * No eigendecompositions or matrix operations required
  * Computational costs O(|V|FF'+|E|F') similar to GCNs
* 2) Different importances to nodes of a same neighborhood
  * interpretability에 유리함 
* 3) Does not depend on upfront access to global graph structure or all of its nodes
  * Inductive learning 가능: including tasks where the model is evaluated on graphs that are completely unseen during training
* 4) Works entirly of the neighborhood and does not assume any ordering within it. 

**유의할점**
* the size of the receptive field of our model is upper-bounded by the depth of the network
* parallelization across all the graph edges may involve a lot of redundant computation as the neighbors will often highly overlap in graphs of interest

## 3. Evaluation
#### 3.1. Datasets
* Transductive learning
  * citation network benchmark datasets -Cora, Citeseer and Pubmed
  * nodes: documents, edges: citations
    * node features: bag-of-words representation of a document
* Inductive learning
  * PPI dataset

#### 3.2. SOTA methods for comparison
* Transductive learning
  * Label propagation, semi-supervised embedding, manifold regularization, skip-gram based embeddings, iterative classification algorithms, and planetoid
  * GCNs, graph convolution models utilizing higher-order Chebyshev filters, MoNet 
* Inductive learning
  * four supervised GraphSAGE inductive methods presented in GraphSAGE-GCN, GraphSAGE-mean, GraphSAGE-LSTM, GraphSAGE-pool

#### 3.4. Results
* Summary result of transductive learning
![image](https://user-images.githubusercontent.com/60350933/150372780-5f22093c-7246-4931-98c7-300d20e45f65.png)
* Summary result of inductive learning
![image](https://user-images.githubusercontent.com/60350933/150374554-4a883c82-2f39-42c1-b303-c17852dddb58.png)

## 4. Conclusions
potential improvements and extensions
* handle large batch sizes
* Attention mechanism to perform analysis on model interpretability
* extending the method to perform graph classification
* Extending the model to incorporate edge features (relationship among nodes)


