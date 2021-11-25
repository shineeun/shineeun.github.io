---
layout: post
title:  "[그래프데이터분석및응용] 네트워크의 Motifs, graphlets and structural roles"
subtitle:   "네트워크 motifs, graphlets and structural roles"
categories: study
tags: graphs
comments: true
header-img:

---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. Motifs

1. **subgraphs**
   * building blocks of networks: discriminate or characterize networks
   * consider all possible (non-isomorphic) directed subgraphs of size 3

2. **graph isomorphism**
   * Graph G와 H는 f:V(G) -> V(H)가 존재하면 isomorphic이라고 한다
   * any two nodes u and v of **G** are adjacent in **G** if f(u) and f(v) are adjacent in H

3. **Network significance profile**
   * A feature vector with values for all subgraph types
   * 같은 network의 도메인들은 similar significant profiles를 가진다. 
   * 
4. **motifs**
   * network가 어떻게 작동하는 지 이해할 수 있게 보조하는 기능
   * 예시
     * Feed-forward loops: 신경망에서 주로 많이 발견이 됨
     * Parallel loops: food webs
     * Single-input modules: gene control networks
      ![image](https://user-images.githubusercontent.com/60350933/143463090-e37c47cd-bf6d-4628-950d-7fb83cf7d50d.png)
      ![image](https://user-images.githubusercontent.com/60350933/143463120-2e7a3dda-47e7-4655-855b-30bac260dc21.png)
      ![image](https://user-images.githubusercontent.com/60350933/143463041-8362a759-5685-4bb5-aa93-1a7af59e6005.png)
  * induced subgraph of interest라고 한다.
    ![image](https://user-images.githubusercontent.com/60350933/143463430-312eb707-ffa7-4b7e-bf76-5d5f4f747e52.png)
  * Allow overlapping of motifs
  * 특정 network가 유의한지를 보기 위해서, random graph보다 motif가 많은지를 확인하고자 한다. 
    $N_ireal$은 number of subgraphs of type i in network $G_real$
    $N_irand$은 number of subgraphs of type i in the randomized network $G_rand$
  * network significance profile (SP)
  
5. **Generation of random graphs** 
  - generate a random graph with a given degree sequence k1, k2, ..., kn
  - Useful as a 'null' model of networks
  - 1. Configuration model 
    ![image](https://user-images.githubusercontent.com/60350933/143464637-fca30292-8ce3-4909-895d-897a158a599e.png)
  - 2. Switching
      * start from a given graph G and repeat the switching step Q*|E| times
        * select a pair of edges A-B, C-D at random

        * exchange the endpoints to give A-D, C-D

          *exchange edges only if no multiple edges or self-edges are generated*

      * A randomly rewired graph as a result

      * Q is chosen large enough for the process to converge

        100 정도..?

6. Variations of the Motif Concept
   * Canonical definition
     * directed and undirected
     * colored and uncolored
     * temporal and static motifs
   * Variations on the concept
     * Different frequency concepts
     * Different significance metrics
     * Under-representation (anti-motifs)
     * Different constraints for null model



## 2. Graphlets

1. 개념

   Connected non-isomorphic subgraphs

2. **Graphlet Degree Vector**

   - Used to obtain a node-level subgraph metric

     *Degree : the number of edges that a node touches* *

   * Generalized notion of 'degree' for graphlets

   * **the number of graphlets that a node touches** 

3. **Automorphism orbit**

   * takes **symmetries of a subgraph** into account 

     고유한 orbit이 무엇인가? 

   * Graphlet Degree vector의 재정의

     A vector with the frequency of the node in each orbit position
     ![image](https://user-images.githubusercontent.com/60350933/143467159-5cc28242-d7fc-4f32-a502-b8e4af594af0.png)

 4. **Finding Motifs and Graphs**

   * Network-centric approaches
     * Enumerating: all k-sized connected subgraphs
     * Counting: the number of occurrences of each subgraph type

   * Computation time grows exponentially as the size of the motif/graphlet increases
   * **Algorithms**
     * **Exact subgraph enumeration (ESU)**
     * Kavosh
     * Subgraph sampling

5. **ESU algorithm**

   * node v에서 시작하여 nodes u를 V_extension set에 아래의 2가지 특성을 가진다면 추가한다
     * u의 node_id가 v의 node_id보다 더 커야한다
     * u는 새로운 node의 w의 이웃일 수도 있지만 V_subgraph에 이미 있는 node와의 이웃은 아니어야 한다. 
   * recursive function의 일종이며, tree-like structure of depth k (ESU-Tree)
    ![image](https://user-images.githubusercontent.com/60350933/143467969-969e7911-d04d-4d86-a47a-2ddb3e56fc1c.png)
   * After enumeration of all k-sized subgraphs, **the graphs should be counted**

## 3. Structural Roles

1. **Roles**

   * measured by structural behaviors

   * A group of nodes with similar structural properties

   * Cf. Communities/groups: A group of nodes that are well-connected to each other

     * Roles and communities are complementary

   * Structural Equivalence

     Nodes u and v are structurally equivalent if they have the same relationships to all other nodes

   * 왜 중요한가?

     * 변화나, outlier, identity resolution

2. **Structural Role Discovery Method**

   * RoIX: Authomatic discovery of nodes' structural roles in networks (Henderson et al., 2011)
   * Neighborhood features
      * Node's connectivity pattern
      * Base set of a node's neighborhood features
        * Local features: all measures of the node degree
        * Egonet features: the number of within-egonet edges, edges entering or leaving egonet

  * Recursive Feature Extraction
    * To what kind of nodes is a node connected?
    * Aggregate features of a node and use them to generate new recursive features
      * set of current node features to generate additional features
      * correlation은 발생할 수 있지만 제외할 수 있음

  * Role Extraction
    Clustering method를 사용할 수 있다. 
