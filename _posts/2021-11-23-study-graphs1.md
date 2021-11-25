---
layout: post
title:  "[그래프데이터분석및응용] 네트워크 데이터의 구조 및 특징"
subtitle:   "네트워크 데이터의 구조 및 특징"
categories: study
tags: graphs
comments: true
use_math: true
header-img:
---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. Structures and Properties of Networks

#### 1. 네트워크를 측정하는 척도

1. **Degree distrubtion: P(k)**

   * Degree: 

     * Out-degree: 특정 node에서 나가는 연결의 개수
     * In-degree: 특정 node로 들어오는 연결의 개수

   * Degree Distribution

     probability that a randomly chosen node has degree k

     * N(k)=number of nodes with degree k 

2. **Path length: h**

   * path: sequence of nodes in which each node is linked to the next one

   * distance (shortest path, geodesic)

     2개의 nodes를 잇는 가장 짧은 path의 edge의 개수

     * 만약에, node 2개가 서로 연결이 되어있지 않다면, 거리는 무한대(혹은 0)로 정의가 된다. 

     * directed graphs의 경우에는 direction에 따라 path가 성립이 되어야 한다. 
       * h(a,b) != h(b,a)

   * Diameter

     The maximum (shortest path) distance between any pair of nodes in a graph

     (연결이 되지 않는 경우는 제외가 됨)

   * Average path length for a <span style="color:blue">connected graph</span> or a <span style="color:blue">strongly connected directed graph</span> 

     * <span style="color:blue">connected graph</span> 

       * connected vertices

         만약 u에서 v까지의 path가 존재하면 2개의 node u,v는 연결이 되어있는 것이다. 

       * 모든 node의 pair이 connected된 경우

       * directed graph의 경우

         * weakly connected: 방향성을 무시했을 때 connected graph인 경우
         * strongly connected: 방향성을 고려했을 때 connected graph인 경우

       * connected component (undirected)

         연결된 node들 간의 부분집합 중 가장 많은 연결이 되어있는 subgraph. 

       * strongly connected component (directed)

         directed graph에서 strongly connected subgraph들 중 가장 큰 subgraph

         *아래 그래프 에서 (a-b-e)가 strongly connected component이다.*

     ![image-20211124001319929](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20211124001319929.png)

3. **Clustering coefficient: C**

   그래프 형성하는 노드들이 얼마나 랜덤하게 생성되어있는가? 

   *내가 아는 두 친구가 서로 아는 확률이 얼마 정도일까?* 

   하나의 node 기준으로 두개의 이웃 node가 연결될 확률이 높다면 랜덤성이 낮다고 볼 수 있다. 
   $ C_i=\frac{2e_i}{k_i(k_i-1)}\,where \, e_i \, is \, the\,  number\, of\, edges\, between\, the\, neighbors\, of\, node\,i\\k_i(k_i-1)\,is\,max\,number\,of\,edges\,between\,the\,k_i\,neighbors$

   * Clustering coefficient는 undefined for nodes with degree 0 or 1

   * Average clustering coefficient

     전체 그래프의 평균
     $C=\frac{1}{N} \displaystyle\sum_{i}^{N}{C_i}$
     

4. **Connected components: s**

   * Size of the largest connected component (=giant connected component)
     - largest set where any two vertices can be joined by a path

## 2. Models to generate realistic networks
---

1. **Erdös-Rényi Random Graph Model**

   링크가 랜덤으로 형성되어 있을 것이다. 

   * Two variants: 

     * G(np): Undirected graph on n nodes where each edge (u,v) appears i.i.d with probability p
     * G(nm): Undirected graph with n nodes, and m edges picked uniformly at random

   * G(np)

     * n과 p가 uniquely determine the graph

     * same n,p를 가지더라도 다른 그래프가 생성이 될 수 있음

     * 특성

       * Degree distribution: binomial

         n이 커짐으로 인해, degree의 평균과 표준편차의 값이 작아질 것이다. 

       * Clustering coefficient
         $E[e_i]=p\frac{k_i(k_i-1)}{2}\\
         E[C_i]=p \approx \frac{kmean}{n|}$

       * Path length
         $h \approx O(logn)$
         

         - 15번 정도면 모든 노드에 전달이 될 수 있다. 

       * Connected component

         * s: Giant connected component exists where kmean>=1

2. **Small World model** (Watts-Strogatz, 1998)

   High clustering과 short paths (small diameter)를 동시에 가지는 방법

   * clustering: edge "locality"
     * low-dimensional regular lattice
   * randomness enables 'shortcuts'
     * add or remove edges to create shortcuts to join remote parts of the lattice

3. **Barabasi-Albert model**

   * Growth: the number of nodes in the network increases over time

   * Preferential Attachment

     more connected a node is, the more likely it is likely to receive new links 

   * 한계

     * diameter이 점점 더 늘어나는 것이 아니라 직경이 더 짧아지는 현상이 발생함

4. **Kronecker Graph model**

   * Recursive graph generation 기반

     * Intuition: self-similarity

       object is similar to a part of itself

   * Kronecker product of matrices A and B

     Kronecker product of two graph를 Kronecker product of their adjacency matrices로 정의

     ![image-20211124005912163](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20211124005912163.png)

   * 반복을 통해 실제 차원의 데이터를 얻을 수 있음

   * Stochastic Kronecker graph

     matrix를 확률로 표현한 것
