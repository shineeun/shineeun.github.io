---
layout: post
title:  "[그래프데이터분석및응용] Community Detection"
subtitle:  "Community Detection"
categories: study
tags: graphs
comments: true
header-img:

---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. Community in networks
#### 1. Communities
   * 서로 긴밀한 node들의 집합으로, 해당 집합 내의 node 간의 연결은 높다

#### 2. Flow of information
   * how close an actor to all other actors in the network?
     * what structurally distinct roles do nodes play?
     * what roles do different **links** play?
   * Two perspective on **links**
     * structural: span different parts of the network
       - **triadic closure**: if two people in a network have a friend in common, then there is an increased likelihood they will become friends themselves
     * interpersonal: either strong or weak

#### 3. Triadic clousre
   * High clustering coefficient로 이어진다. 
   * Reasons
     * If B and C have a friend A in common then, 
       - B is more likely to meet C as they both spend time with A
       - B and C trust each other since they have a friend in common
       - A has incentive to bring B and C together. 

#### 4. Graovetter's Explanation
    * 아주 가까운 지인보다 가끔 만나는 사람으로부터 직장에 대한 정보를 얻을 수 있음. 
    * a connection between the social and structural role of an edge
    * **Structure** 
      * structurally embedded edges are also socially strong
      * Long-range edges spanning different parts of the network are socially weak. 
    * **Information**
      * Long-range edges allow you to gather information from different parts of the network and get a job
      * Structurally embedded edges are heavily redundant in terms of information access. 

#### 5. Network Communities
   * Communities: set of tightly connected nodes
   * **Modularity Q**
     * A measure of how well a network is partitioned into communities
     * Given a partitioning of the network into groups disjoint s, 
       ![image](https://user-images.githubusercontent.com/60350933/148061143-98adccef-5138-42bb-bf19-258e256f4f54.png)
   * Null model: configuration model
     * Given real G on *n* nodes and *m* edges, construct rewired network G'
       - Same degree distribution but uniformly random connections
       - Consider G' as a multigraph
       - The expected number of edges between nodes i and j of degrees k(i) and K(j) equals:
         ![image](https://user-images.githubusercontent.com/60350933/148061347-c5548a5b-eaee-43cb-aba3-5723055f7a73.png)
       - The expected number of edges in multigraph G' (위의 식이 맞는 지 검토하는 방법)
          ![image](https://user-images.githubusercontent.com/60350933/148061420-384a1e94-01f2-48d7-9fb7-db289a2dee12.png)
    * Q = (fraction of edges that connect nodes of the same group) - (fraction of such edges if the edges were positioned at random)
      ![image](https://user-images.githubusercontent.com/60350933/148061626-8969f2f2-6402-4da3-8660-2b3e97549986.png)
      * 2m으로 나눠준 것은 왕복을 고려한 것
    * Modularity values take range [-1,1]
      * it is positive if the number of edges within groups exceeds the expected number
      * 음수이면 anti-community (적대관계)라고 간주함. 
      * **Q greater than 0.3-0.7 means significant community structure**
    
## 2. Louvain Algorithm (Non-overlapping communities)
#### 1.Overview
   * Greedy algorithm for community detection
   * O(nlogn) run time
   * Supports weighted graphs 
   * provide hierarchical communities
   * High level
     * Greedily maximizes modularity
     * Each pass is made of 2 phases
       - Phase 1: modularity is optimized by allowing only local changes to node-communities memberships
       - Phase 2: The identified communities are aggregated into super-nodes to build a new network
     * The passes are repeated iteratively unil no increase of modularity is possible.  
  
#### 2. 1st phase
   * Put each node in a graph into a distinct community
   * For each node i, the algorithm performs two calculations
     * compute the modularity delta when putting node i into the community of some neighbor j
     * Move i into a community of node j that yeilds the largest gain in delta Q. 
    * Phase 1 runs until no movement yields a gain. 
    * 어느 node부터 볼 것인지는 중요하지 않아서 random으로 선택할 수 있다. 
   
#### 3. 2nd phase (restructuring)
   * The communities obtained in the first phase are contracted into super-nodes, and the network is created accordingly
     - super-nodes are connected if there is at least one edge between the nodes of the corresponding communities
     - The weight of the edge between the two super-nodes is the sum of the weights from all edges between their corresponding communities. 
     - Phase 1 is then run on the super-node network. 


## 3. Overlapping communities
#### 1. BigCLAM
   * one of the methods to detect overlapping communities using MLE. 
 
     
