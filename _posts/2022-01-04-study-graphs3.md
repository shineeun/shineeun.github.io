---
layout: post
title:  "[그래프데이터분석및응용] Network centrality"
subtitle:   "네트워크 중심성"
categories: study
tags: graphs
comments: true
header-img:

---

해당 포스팅은 [연세대학교 21-1 UT 세미나 그래프데이터분석과응용]을 기반으로 작성되었습니다. 

## 1. Network Centrality in indirected networks

1. **Degree Centrality**
   *  The number of nearest neighbors
   *  Normalized degree centrality: 노드가 많아지면, nearest neighbor도 많아지기 때문에, 정규화 해주는 것 (0-1 scale로 표시)
   *  High Degree를 가진 다는 것은 Direct contact with many other actors. 

2. **Closeness centrality**
   * How close an actor to all the other actors in network. 
     ![image](https://user-images.githubusercontent.com/60350933/148014520-92614dad-5306-487b-905f-9595612515de.png)
   * Normalized closeness centrality
     ![image](https://user-images.githubusercontent.com/60350933/148014624-54e280a0-48fe-4498-b87c-87895d831638.png)
   * High closeness를 가진다는 것은 다른 node들까지의 거리가 가장 짧은 경우로, all other nodes로 가기 위한 step이 가장 짧다는 것을 의미한다. 
   * Disconnected network
      * i에서 j까지 이어지는 경로가 없을 때는 무한대로 계산을 하며, centrality 계산 시에 제외를 해준다.
      * Harmonic centrality를 계산을 할 수도 있음
        ![image](https://user-images.githubusercontent.com/60350933/148014763-13e04fac-ac61-440c-8098-1ed045033b32.png)

3. **Betweenness Centrality**
   * The number of shortest paths going through the node i
     ![image](https://user-images.githubusercontent.com/60350933/148015187-95c34b23-bde9-49ba-9fb8-50f674cecf01.png)
      * s와 t를 이어주는 최단 경로들 중에서 i를 지나는 최단 경로의 수
   * Normalized betweenness centrality
     ![image](https://user-images.githubusercontent.com/60350933/148015265-22f2a58f-ca3c-4d15-9984-ede1c985dc85.png)
   * High betweenness를 가진다는 것은 vertex lies on many shortest paths. 
   
4. **Eigenvector Centrality**
   * 어떤 이웃인지를 고려하는 것
     ![image](https://user-images.githubusercontent.com/60350933/148023103-fa41d3d4-7fe4-4770-8d33-7c8683638987.png)
     ![image](https://user-images.githubusercontent.com/60350933/148024084-5c03b46c-98b3-4a15-a986-6645de69ae07.png)
     * select an eigenvector associated with the largest eigenvalue
     * Perron-Frobenius theorem에 의함
       - A real square matrix with positive entries has a unique largest absolute real eigenvalue, leading eigenvalue
       - The corresponding eigenvector can be chosen to have strictly positive components, leading eigenvector
       - The other eigenvectors have at least one negative or non-real component. 
      * 이를 활용하여 node 수만큼의 eigenvector 중 leading eigenvector를 선택하는 것
   * High eigenvector를 가지면 vertex가 중요한 이웃을 가짐을 알 수 있다. 

## 2. Network centralities in directed networks

1. **In and out-degree centrality** 
   ![image](https://user-images.githubusercontent.com/60350933/148025581-deec863b-71af-4bf6-b88f-4b29a78a6577.png)

2. **Closeness centrality**
   ![image](https://user-images.githubusercontent.com/60350933/148025621-486adc01-5064-4e54-923a-7c5e7f20ef90.png)

3. **Betweenness centrality**
   ![image](https://user-images.githubusercontent.com/60350933/148025681-1d9031ab-079a-487f-8a78-133dbfe08a66.png)
   
 4. **PageRank**
    * If the adjacency matrix is asymmetric, the eigenvector centrality is useless. 
    * A 'vote' from an important page is worth more
      * Each link's vote is proportional to the importance of its source page
      * If page i with importance r(i) has d(i) out links, each links gets r(i)/d(i) votes
      * Page j's own importance r(j) is the sum of the votes on its in-links 
    
    * A page is important if it is pointed to by other important pages
      * Define a 'rank' r(j) for node j  
        ![image](https://user-images.githubusercontent.com/60350933/148026723-49b8a4ec-139d-4c8f-adea-087f124f8cb3.png)
        ![image](https://user-images.githubusercontent.com/60350933/148026819-c5576a31-3190-4250-9ebb-6c4f055309e2.png)
        - node가 많아지면 Gaussian elimination으로 해를 구할 수가 없을 것이다. 
    
    * Matrix formulation
      * Stochastic adjacency matrix M
        - Let page j have d(j) out-links
        - If j-> i, then M(ij)= 1/d(j)
      *  M is a column stochastic matrix where each column sums to 1. 
      *  Rank vector r: an entry per page
         - r(i) is the importance score of page i
         - sigma(r(i))=1
      *  The flow equations
         - r=M*r
         - rank vector r을 찾아야 함
    
    * Eigenvector Formulation
      * Rank vector r is an eigenvector of the stochastic matrix M. 
      * Power iteration: efficient solution to obtain r
        - produces the leading eigenvalue and its corresponding eigenvector
        - A simple iterative scheme (약 50번 정도의 iteration을 거치면 limiting solution을 얻을 수 있음)
    
    * Rankdom walk interpretation
      * Imagine a random web surfer
        - At any time t, surfer is on some page i
        - At time t+1, the surfer follows an out-link from i uniformly at random.
        - Ends up on some page j linked from i
        - Process repeats indefinitely
      * Let P(t): vector whose i-th coordinate is the probability that the surfer is at page i at time t.
        - P(t)는 probability distribution over pages
      * Where is the surfer at time t+1?
        - Follow a link uniformly at random: p(t+1) = Mp(t)
      * Suppose the random walk reaches a state p(t+1)=Mp(t)=p(t)
      * Our original rank vector r satisfies r=Mr
        - r is a stationary distribution for the random walk
      
    * Does this converge?
      * Problem 1: Spider trap
        - PageRank scores are not what we want
        - Solution: teleport를 사용하여 beta의 확률로 link를 따라가거나 (1-Beta)의 확률로 random page로 이동
      * Problem 2: Dead end
        - The matrix is not column stochastic (out link가 없으므로, 어느 열에서도 transition될 확률이 0이 됨)
        - solution: teleport를 사용하여 dead end의 경우 1의 확률로 teleport를 시키는 것 (adjust matrix accordingly)
