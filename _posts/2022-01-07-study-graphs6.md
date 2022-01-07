---
layout: post
title:  "[그래프데이터분석및응용] Graph Neural networks"
subtitle:  "Graph deep learning"
categories: study
tags: graphs
comments: true
header-img:

---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. Graph Neural Networks
#### 1. Deep Graph Encoders
   * multiple layers of non-linear transformations based on graph structure 
   * Output: node embeddings, subgraph or graph embeddings
   * Difficulties
     * No fixed node ordering or reference point
     * Dynamic and have multimodal features
   * Local network neiborhoods
   * Stacking multiple layers 
   
#### 2. Setup
   * V: vertex set
   * A: adjacency matrix (assume binary)
   * X: matrix of node features
   * *v*: node in V; N(v): the set of neighbors of v
   * Node features
     * Social network의 경우 user profile, user image
     * biologial networks: gene expression profiles, gene function information
     * When there is no node feature in the graph dataset
       * Indicator vectors (one-hot encoding of a node)
       * Vector of constant 1: [1,1,...,1]

#### 3. Naive approach
   * Join adjacency matrix and features
   * Feed them into a deep neural net
     * Issues with this idea
       - O(|v|) parameters
       - Not applicable to graphs of different sizes
         : input만 달라져도 새로운 신경만을 학습을 시켜줘야 한다. 
       - Sensitive to node ordering
         
         
#### 4. Convolutional networks
   * Goal: to generate convolutions beyond simple lattices
   * Leverage node features/attributes
   * From Images to graphs
     * Transform information at the neighbors and combine it
       - Transform 'messages' *h(i)* from neighbors: W(i)*h(i)
       - Add them up

#### 5. Graph Convolutional networks
   * Nodes' neighborhood defines a computation graph
   * Learn how to propagate information across the graph to compute node features
   * **Local network neighborhoodsd**
     - Aggregate neighbors
     - generate node embeddings based on local network neighborhoods
     - Every node defines a computation graph based on its neighborhood
       ![image](https://user-images.githubusercontent.com/60350933/148484300-b8178195-37ce-4f8a-8129-a6509ace5a30.png)
   * Many layers 
     * Models can be of arbitrary depth
       - Nodes have embeddings at each layer
       - Layer 0 embedding of node u is its input features, x_u.
       - Layer k embedding gets information from nodes that are k hops away. 
   * Neighborhood Aggregation
     * Key distinctions are in how different approaches aggregate information across the layers
     * Average information from neighbors -> apply a NN
       ![image](https://user-images.githubusercontent.com/60350933/148484594-4e61f225-4337-4e32-8605-f66cc149e4b9.png)
   * Model parameters
     ![image](https://user-images.githubusercontent.com/60350933/148484823-899f1349-c848-41a3-9149-ac4a5b43562c.png)
   * Matrix Formulation 
     ![image](https://user-images.githubusercontent.com/60350933/148485124-6f962785-c3fe-4f6d-8be5-ccebf312e654.png)
     - not all graphs can be represented in the matrix form
   * How to train GNN
     * Node embedding is a function of input graph
     * Supervised Setting: minimize the loss
       - Directly train the model for the supervised task
         ![image](https://user-images.githubusercontent.com/60350933/148485430-104de7ab-9c0a-4f4b-8a0e-15b3df0fd32d.png)
     * Unsupervised setting
       - use the graph structure as the supervision
       - Similar nodes have similar embeddings
         ![image](https://user-images.githubusercontent.com/60350933/148485365-53496b4f-a024-4972-930e-699e4fcc7e35.png)
       - node similarity can be anything including random walks or node proximity in graph
   * 가장 큰 장점: Inductive capability
     * Generate embeddings for nodes as neede and even for nodes we never trained on. 
     * The same aggregation parameters are shared for all nodes
     * The number of model parameters in sublinear in |v| and can generalize to unseen nodes
     * Number of parameters to train does not differ by the node. 

#### 6. General perspectives on GNN
   * GNN layer=message+aggregation
     * GCN, GraphSAGE, GAT: how to design the message and define the function to aggregate the messages
   * A single GNN Layer
     * Compress a set of vectors into a single vector
     * Two step process
       - Message
       - Aggregation
   * 1) Message Computation
     * Message function
       ![image](https://user-images.githubusercontent.com/60350933/148486041-ef2071b5-f8ee-4840-be28-2b4b054c5485.png)
     * Intuition: Each node will create a message, which will be sent to other nodes later
   * 2) Aggregation
     * Intuition: each node will aggregate the message from node v's neighbors
       ![image](https://user-images.githubusercontent.com/60350933/148486134-8faec46e-dd40-47b1-b38f-a91a4edc283e.png)
   * Issue
     * information from node v itself could get lost
     * computation of h_v(l) does not directly depend on h_v(l-1)
     * solution: include h_v(l-1) when computing h_v(l)
     * Message에서의 해결방법: compute message from node v itself
       - a different message computation will be performed
     * Aggregation의 해결방법: after aggregating from neighbors, we can aggregate the message from node v itself via concatenation or summation
 
 #### 7. Classical GNN Layers: GCN
    * GCN
      ![image](https://user-images.githubusercontent.com/60350933/148486650-ebc3bda5-65d3-4aff-bfef-a748918b91c2.png)
    * Message
      ![image](https://user-images.githubusercontent.com/60350933/148486676-805fa702-9aba-4e5b-9f8f-3ddcee4e00fa.png)
    * Aggregation: sum over message from neighbors, then apply activation function. 
      ![image](https://user-images.githubusercontent.com/60350933/148486698-1606e7f6-fa6f-41b6-a3b2-05884e090363.png)

#### 8. GraphSAGE
   * GraphSAGE (Sample and aggreGatE)
     ![image](https://user-images.githubusercontent.com/60350933/148486804-b6378ff0-d557-4d8e-b673-da5efe6752e9.png)
   * Message is computed within the AGG. 
   * Two-stage aggregation
     * Stage1: Aggregate from node neighbors
       ![image](https://user-images.githubusercontent.com/60350933/148486867-880a1a8b-8a06-49f1-ab1c-40fdc31d356c.png)
     * Stage2: Further aggregate over the node itself
       ![image](https://user-images.githubusercontent.com/60350933/148486883-14f76721-7994-421f-ae14-c20c713ed6ff.png)
   * Neighbor Aggregation
     * Mean: take a weighted average of neighbors
       ![image](https://user-images.githubusercontent.com/60350933/148486967-8041fe23-7c89-4981-a4d7-d71a88022b26.png)
     * Pool: transform neighbor vectors and apply symmetric vector function Mean, Max, sum, or std. 
       ![image](https://user-images.githubusercontent.com/60350933/148486990-cb023865-312a-443b-a033-9387d6ef0e7d.png)
     * LSTM  
   * L2 Normalization
     * Apply l2 normalization to h_v(l) at every layer: 크기가 1인 벡터로 정규화
       ![image](https://user-images.githubusercontent.com/60350933/148487068-cb8329b5-37d9-4749-80fb-9023ff31d0af.png)
     * Without l2 normalization, the embedding vectors have different scales for vectors
   * Neighborhood sampling
     * Randomly sample a node's neighborhood for message passing
     * When we compute the embeddings, we can sample different neighbors
     * greatly reduces computational cost and avoids overfitting
   * GCN이나 GraphSAGE의 경우 이웃들에게 공통된 weight을 부여함
     * All neighbors are almost equally important to node v

#### 9. Graph Attention Networks (GAT)
   * Attention weights를 부여
     ![image](https://user-images.githubusercontent.com/60350933/148487503-b1bb5022-e7de-4049-a489-1f9e21a50bd3.png)
   * Not all node's neighbors are equally important and attention focuses on the important parts of the input data and fades out the rest
   * Idea: more computing power on that small but important part of the data
   * Goal: Specify arbitrary importance to different neighbors of each node in the graph
   * How?
     * Compute embedding of each node in the graph following an attention strategy
   * Attention mechanism
     * Let *a* compute attention coefficients *e_vu* across pairs of nodes *u*, *v* based on their messages
       ![image](https://user-images.githubusercontent.com/60350933/148487740-e9a4fa45-ee55-434d-b9ed-935c4c60f473.png)
     * indicates the importance of u's message to node v
     * Normalize e_vu into the final attention weight
     * Use the softmax function, so that
       ![image](https://user-images.githubusercontent.com/60350933/148487860-643caf49-da5b-4e84-aef1-cc37ae555d59.png)
       ![image](https://user-images.githubusercontent.com/60350933/148487871-ae5cacf9-4c0b-4ab3-86e4-5329b2cd793a.png)
     * Weighted sum based on the final attention weight
       ![image](https://user-images.githubusercontent.com/60350933/148487908-b9c88881-0d60-4638-87fa-3b1b44895d5b.png)
     * What is the form of attention mechanism *a*?
       * agnostic to the choice of *a*
       * use a simple single-layer NN as *a* and *a* has trainable parameters
       * Parameters of *a* are trained jointly with weight matrices in an end-to-end fashion
   * Multihead attention
     * Allows model to capture different version of attention
     * Stabilizes the learning process of attention mechanism
     * Create multiple attention scores (each replica with a different set of parameters)
       ![image](https://user-images.githubusercontent.com/60350933/148488198-f29c640c-70dc-4192-b36a-024cb99ac606.png)
     * Outputs are aggregated by concatenation or summation. 

#### 10. Oversmoothing problem in GNN
   * Overview
     * The issue of stacking many GNN layers
     * All node embeddings converge to the same value
    
   * Receptive Field
     * the set of nodes that determine the embedding of the node of interest
     * In the k-layer GNN, each node has receptive field of k-hop neighborhood
     * Receptive field overlap for the two nodes and the shared neighbors quickly grows when we increase the number of hops (in other words: number of GNN layers)
     * If two nodes have highly-overlapped receptive fields, then thier embeddings are highly similar
     * If GNN layers are many-> node embeddings will be highly similar and suffer from the over-smoothing problem
   
   * Design GNN laer connectivity
     * Lesson1: Be cautious when adding GNN layers
       - Step 1: Analyze the necessary receptive field to solve your problem
       - Step 2: Set number of GNN layers to be a bit more than the receptive field we like
       - Do not set L unnecessarily high
       - 
   * Expressive power for shallow GNN
     * How to make a shallow GNN more expressive
     * Idea1
       * Increase the expressive power within each GNN layers
       * Make aggregation/transformation become a deep NN. 
     * Idea 2
       * A GNN does not necessarily only contain GNN layers
       * We can add MLP layers before and after GNN layers, as pre- and post-process layers 
       ![image](https://user-images.githubusercontent.com/60350933/148488744-a07f554b-fb71-4418-8972-c61e06e14b6b.png)


