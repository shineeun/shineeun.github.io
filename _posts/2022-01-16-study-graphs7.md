---
layout: post
title:  "[논문리뷰] A survey on knowledge graphs: representation, acquisition and application "
subtitle:  "Knowledge Graph"
categories: study
tags: graphs
comments: true
header-img:

---

Ji, S., Pan, S., Cambria, E., Marttinen, P., & Philip, S. Y. (2021). A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on Neural Networks and Learning Systems.

## Abstract
Human knowledge provides a formal understanding of the world. Knowledge graphs that represent structuralrelations between entities have become an increasingly popular research direction towards cognition and human-level intelligence. In this survey, we provide a comprehensive review of knowledge graph covering overall research topics about 1) knowledge graph representation learning, 2) knowledge acquisition and completion, 3) temporal knowledge graph, and 4) knowledge-aware applications, and summarize recent breakthroughs and perspective directions to facilitate future research. We propose a full-view categorization and new taxonomies on these topics. Knowledge graph embedding is organized from four aspects of representation space, scoring function, encoding models, and auxiliary information. For knowledge acquisition, especially knowledge graph completion, embedding methods, path inference, and logical rule reasoning, are reviewed. We further explore several emerging topics, including meta relational learning, commonsense reasoning, and temporal knowledge graphs. To facilitate future research on knowledge graphs, we also provide a curated collection of datasets and open-source libraries on different tasks. In the end, we have a thorough outlook on several promising research directions.

## 1. Overview
### A. Brief history of knowledge bases
   * Reasoning and problem solving에 많이 사용되어 왔으며, resource description framework (RDF)와 Web Ontology Language (OWL)을 통해서 Semantic web에 대한 중요한 기준들이 세워졌다. 
   * 이를 기반으로, WordNet, DBpedia, YAGO, Freebase와 같은 knowledge bases와 ontologies가 published 됨
   * Knowledge Vault를 기반으로 한 Google의 search의 출시로 knowledge graph의 개념이 널리 알려졌다. 

### B. Definitions
   * **Knowledge graph**
     * ontology로부터 정보를 얻고 ontology에 정보를 통합하며, 새로운 지식을 얻기 위해 reasoner을 apply하는 것
     * Entity(node)와 relation(different types of edges)으로 구성된 multi-relational graph. 
     * G = {E, R, F}
       * E= entities, R= relations, F=facts
       * (h, r, t) ∈ F
       
   * TABLE 1. Notations and descriptions
     ![image](https://user-images.githubusercontent.com/60350933/149667735-25e3bb91-02d4-438c-ae15-366594a5e1c3.png)
   
### C. Knowledge graph 연구 분류
   * Figures 2. Categorization of research on knowledge graphs
     ![image](https://user-images.githubusercontent.com/60350933/149667760-bb34abae-f7e5-4cd3-8bc4-884823a3a3be.png)

#### 1) Knowledge Representation Learning
Representation learning은 다음과 같은 4가지의 측면으로 볼 수 있다. 
* **Representation space** : relations과 entities가 represented됨
  - point-wise space, manifold, complex vector space, Gaussian distribution, and discrete space
* **Scoring Function** : factual triples의 plausibility를 측정
  - distance-based와 scoring function을 기반으로한 similarity matching으로 나뉨. 
* **Encoding models** : relational interaction을 representing하고 learning
  - 최근 연구가 많이 집중되고 있는 분야
  - linear/bilinear models, factorization, neural network 등이 있다. 
* **Auxiliary information** : 외부 정보 활용
  - textual, visual and type information을 고려함 
  
#### 2) Knowledge Acquisition
* **KGC**
  - expanding existing knowledge graphs 
  - 주요 카테고리: embedded-based ranking, relation path reasoning, rule-based reasoning, and meta relational reasoning
* **relation extraction**
  - discover new knowledge from the text
  - 주요 카테고리: recognition, disambiguation, typing, and alignment
  - attention mechanism, graph convolutional networks (GCN), adversarial training, reinforcement learning, deep residual learning, transfer learning 등을 주로 사용함.
* **entity discovery**
  - discover new knowledge from the text

#### 3) Temporal knowledge graphs
* Temporal 정보를 representation learning에 사용함
* 주요 카테고리: Temporal embedding, entity dynamics, temporal relational dependency, temporal logical reasoning

#### 4) Knowledge-aware Applications
* 주요 카테고리: Natural language understanding, Question answering (QA), recommendation system 등

## Knowledge representation learning (KRL, KGE, multi-relation learning, statistical relational learning)
KRL 모델을 학습하는 전략은 Appendix D에 소개됨
### A. Representation space
Key Issue: to learn low-dimensional distributed embedding of entities and relations.
* Embedding space should follow three conditions: differentiability, calculation possibility, and definability of a scoring function. 
#### 1) Point-wise space
* TransE: represents entities and relations in d-dimension vector space and makes embedding follow the translational princible
* TransR: introduces separated spaces for entities and relations. 
  * Entities를 relation으로 projection matrix(Mr ∈ Rk×d)를 통해서 project 
* NTN: models entities across multiple dimensions by a bilinear tensor neural layer. The relational interaction between head and tail is captured as tensor
* HAKE: captures semantic hierarchies by mapping entities into polar coordinate system
#### 2) Complex vector space
e h, t, r ∈ C_d
head entity= Re(h))+iIm(h) whree Re(h) is the real part and IM(h) is the imaginary part
* ComplEx, RotatE, QuatE
#### 3) Gaussian distribution
* Gaussian word embedding의 영향을 받아, entities와 relations의 uncertainty를 gaussian distribution을 통해 다룬다. 
* KG2E, TransG
#### 4) Manifold and group
* Manifold space
  * Manifold: topological space, which is defined as a set of points with neighborhoods by the set theory. 
  * The group: algebraic structures defined in abstract algebra
  * Point-wise modeling의 경우 ill-posed algebraic system임 (number of scoring equations is far more than the number of entities and relations)
  * Embeddings are restricted in an overstrict geometric form even in some methods with subspace projection. 
  * ManifoldE, MuRP (represents multi-relational KG in Poincare ball of hyperbolic space)
* Lie group 
  * TorusE: n-dimensional torus space (compact Lie group)에 embedding함으로 TransE의 regularization 문제를 해결함
* dihedral group
  * DihEdral: dihedral symmetry group preserving a 2-dimensional polygon. 
  
### B. Scoring function
* To measure the plausibility of facts (energy function in the energy-based learning framework)
  * Energy-based learning: Energy function Eθ(x)을 학습하고, negative sample보다 positive sample이 더 높은 score을 가질 수 있도록 하는 것. 
* 1) Distance-based와 2) similarity-based function으로 나뉨. 
  ![image](https://user-images.githubusercontent.com/60350933/150673365-9b70dec6-4020-4ac8-ad71-da33deb65432.png)
* 1) Distance-based scoring function
  * Measure the plausibility of facts by calculating the distance between entities, where addictive translatoin with relations as h+r≈ t is widely used
    * Euclidean distance를 measure
  * Structural embedding (SE) uses two projection matrices and L1 distance to learn structural embedding
  ![image](https://user-images.githubusercontent.com/60350933/150679867-7227045b-3243-440b-82a5-541ec08a3a67.png)

  *  TransE -> TransH, TransR, TransA, ITransF
  *  KG2E, ManifoldE
* 2) Semantic-based scoring function
  * semantic matching
  * usually adopts a multiplicative formulation to transform head entity near the tial in the representation space
  * SME (semantically match separate combinations of entity-relation pairs of (h,r) and (r,t))
  * HolE -> HolEx, ComplEx, DistMult
  * ANALOGY
  * Manifold and group models: TorusE, DihEdral
  
### C. Encoding models
Encode the interactions of entities and relations through specific model architectures, including linear/bilinear models, factorization models, and neural networks
* Linear models: formulate relations as linear/bilinear mapping by projecting head entities into a representation space close to tail entities
* Factorization : aims to decompose relational data into low-rank matrices for representation learning
* Neural network: encode relational data with non-linear neural activation and more complex network structures by matching semantic similarity of entities and relations. 
  * Multi-layer perceptron (MLP), neural tensor network (NTN), neural association model (NAM)
* Convolutional Neural networks
  * utilized for learning deep expressive features
  * ConvE: uses 2D convoluation over embeddings and multiple layers of nonlinear features to model the interaction by reshaping head entity and relation into 2D matrix. 
  * ConvKB: CNN for encoding the concatenation of entities and relations witout reshaping.  Keeps transitional characteristics and shows better experimental performance
  * HypER: hypernetwork H for 1D relation-specific convolutional filter generation to achieve multi-task knowledge sharing
* Recurrent neural network 
  * MLP와 CNN 기반 모델은 triplet-level presentation을 학습하는 반면, recurrent networks는 long-term relational dependencies in KG. 
  * RSN: recurrent sip mechanism to enhance semantic representation by distinguishing relations and entities where its relational path is generated by random walk. 
* Transformers
  * Contextualized text representation learning
  * CoKE: employs transformers to encode edges and path sequences.   
  * KG-BERT: BERT model을 사용하여 entities와 relationship encoding
* Graph neural networks (GNN)
  * Learn connectivity structure under encoder-decoder framework
  * R-GCN: relation-specific transformation to model the directed nature of KG
    * GCN은 그래프 인코더 역할을 함
    * Takes the neighborhood of each entity equally
  * SACN: weighted GCN which takes the strength of two adjacent nodes within the same relation type, to capture the structural information in KG by utilizing node structure, node attributes, and relationship types.
  * CompGCN: entity-relation composition operations over each edge in the neighborhood of a central node. 
  
### D. ausiliary information
Multimodal embedding incorporates external information (text 설명, type 제한, relational paths, and visual information), with a KG itself to faciliate more effective knowledge representation
* Textual description
  * KRL: structure and unstructured textual description을 하나의 space에 표현
  * DKRL, SSP (semantic space로의 확장), SSP (embedding-specific loss와 topic-specific loss를 두 개의 목적함수로 사용)
* Type information
  * 일반적으로는 계층적으로 표현함
  * SSE: semantic categories of entities to embed entities belonging to the same category smoothly in semantic space
  * TKRL, KREAR
* Visual information
  * Image-embodied IKRL: image와 structural textual description을 포함
* Uncertain information
  * Uncertain embedding models는 relational facts의 likelihood를 표현하기 위한 uncertainty를 파악하고자 한다. 
  * ProBase, NELL, ConceptNet은 uncertain information with a confidence score assinged to every relational fact
  * Probability calibration 에 대한 연구 진행되고 있음

## 4. Knowledge acquisition
Aims to construct KG from unstructured text and other structured or semi-structured sources, complete an existing KG and discover and recognize entities and relations. 
### A. Knowledge graph completion
Developed to add new triple to a KG
* subtasks: link prediction, entity prediction, and relation prediction
* Focused on learning low-dimensional embedding for triple prediction (embedding-methods)
  * Fail to capture multi-step relations
  * **relation path inference**와 **Rule based reasoning**으로 발전
1) Embedding-based models 
  * Representing inputs and candidates in the unified embedding space
    * KRL 방법론 (TransE, TransH, TransR, HolE, R-GCN), joint learning 방법론 (DKRL with textual information)
  * ProjE : known entities and candidate entities in separately
  * SENN : unified neural shared embedding with adaptively weighted general loss function to learn different latent features
  * ConMask: relationshp-dependent content making over the entity description to select relevant snippets of given relations. 
  * REMEDY: medical domain, conditional relationship variational autoencoder
2) Relation path reasoning:
  * leverage path information over the graph structure
  * Random walk methods
    * Path-Ranking algorithm (PRA), Neural multi-hop relational path modeling, Chain-of-Reasoning, DIVA (unified variational inference freamework that takes multi-hop reasoning as two sub-steps of path-finding and reasoning) 
3) RL-based path filtering
  * Deep reinforcement learning (RL) is introduced for multi-hop reasoning by formulating path-finding between entity pairs as sequential decision making
  * DeepPath, MINERVA, M-Walk
4) Rule-based reasoning
  * To better make use of the symbolic nature of knowledge
  * A rule is defined by the head and body in the form of head <- body
    * Head is an atom (a fact with variable subjects or objects)
    * Body is a set of atoms
  * Logial rules can be extracted by rule mining tools like AMIE, RLvLR
  * 최근 연구 동향
    * Injecting logical rules into embeddings to improve reasoning, with joint learning or iterative training applied to incorporate first-order logic rules
      * KALE: 
        propses a unified joint model with t-norm fuzzy logical connectives defined for compatible triples and logical rules embedding. Three compositions of logical conjuction, disjunction, and negation are defined to compose a truth value of complex formula    
      * RUGE
        iterative model where soft rules are tuilized for soft label prediction from unlabeled triples and labeled triples for embedding rectification
      * IterE
  * Logical rule은 사전 지식을 포함할 수 있어서 multi-hop reasoning을 해석할 수 있고, few-shot labeled relational triples만 가지고도 generalization이 용이하다. 
  * 그러나, logic rules는 한정된 relational facts만 커버할 수 있고 colossal search space에 위험이 있다.  
  * Imcorporating rule based learning for knowledge representation is principally to add regularizations or constraints to representations
    * Neural Theorem Provers (NTP), NeuralLP, Neural-Num-LP, pLogicNet, ExpressGNN
5) Meta relational learning
  * predict new relational facts with only a very few samples
  * GMatching, Meta-KGR, MetaR, Graph Extrapolation networks
6) Triple classification
  * Whether facts are correct
### B. Entity discovery
1) Entity recognition (NER)
   - tags entities in text
2) Entity typing
  - coarse and finegrained types (tree-structured type category and is typiclaly regarded as multi-class and multi-label classification)
3) Entity disambiguation (entity linking)
  - links entity mentions to the corresponding entities
  - 'Einstent won the nobel prize in physics'의 아인슈타인을 Entity 'Albert Einstein'과 연결
5) Entity alignment
  * fuse knowledge among various knowledge graphs 
  * Embedding-based alignment
    * Calculates the similarity between embeddings of pair of entities 
### C. Relation Extraction
- To build large-scale KG automatically by extracting unknown relational facts from plain text and adding them into KG. 
- Distant supervision (weak or self supervision) uses heuristic matching to create training data by assuming that sentences containing the same entity mentions may express the same relaiton under the supervision of relational database. 
1) Neuarl Relation Extraction
2) Attention mechanism
3) GCNs
4) Adversarial training
5) Reinforcement learning 
6) Joint entity and relation extraction

## 5. Temporal Knowledge Graph
## 6. Knowledge Aware applications
A. Language representation learning
* Integrating knowledge into language representatoin
* KG language model (KGLM): render knowledge by selecting and copying entities
* K-BERT: infuses domain knowledge into BERT contextual encoder
* KEPLER: combines knowledge embedding and masked language modeling losses via joint optimization
* GLM: graph-guided entity masking scheme to utilize KG implicitly
* CoLAKE: knowledge context of an entity through a unified word-knowledge graph and modified Transformer encoder

B. Question Answering

C. Recommender systems
KG에서의 multi-hop neighbors의 embedding propagation을 통해 해석 정도를 높인다. 
* CKE, DKN, MKR (multi-task KG representation and recommendation by sharing latent features and modeling high-order item-entity interaction)
* KPRN: interaction between users and items as an entity-relation path in KG
* PGPR: reinforcement policy-guided path reasoning over KG based user-item interaction
* KGAT: GAT를 통해 collaborative knowledge graph of entity-relation and user-item graphs to encode high-order connectivities via embedding propagation and attention-based aggregation. 

## 7. Future dierctions
### A. Complex reasoning
Numerical computing for knowledge representation and reasoning requires a continuous vector space to capture semantic of entities and relations
* up and coming for handling complex reasoning
  * Recurrent relational path encoding
  * GNN-based message passing over KG
  * reinforcement learning based path finding and reasoning
  * Enbabling probablistic inference for capturing the uncertainty and domain knowledge with efficiency embedding will be a noteworthy research question

### B. Unified framework
* HolE 와 ComplEx는 특정 contraint를 가질 때 link prediction에 있어 수학적으로 동일하다 
* KRL 모델 내, joint learning framework 내 유사한 모델들을 unified 할 수 있다. 
* An investigation towards unification in a way similar to the unified framework of graph networks

### C. Interpretability
* 실제 응용에 있어서 굉장히 중요함
* ITransF: sparse vectors for knowledge transferring and interprets with attention visualization
* CrossE: embedding-based path searching to generate explanations for link prediction
* 최근 모델들은 성능은 향상되었지만 interpretability가 낮음 

### D. Scalability
* 컴퓨팅 효율성과 모델 expressiveness간의 trade-off 존재 
  * Rules in a recent neural logical model are generated by simple brute-force search-> large-scale에 적용하기 어렵다
  * ExpressGNN

### E. Knowledge aggregation
* 추천시스템에서 KG를 user-item interaction and text classification을 text와 KG를 semantic space로 encoding하기 위해 같이 사용한다. 
  * 대부분의 Knowledge aggregation methods design neural architectures such as attention mechanisms and GNNs.
  * Pre-trained language model can acquire certain factual knowledge
* Rethinking the way of knowledge aggregation in an efficient and interpretable manner도 중요하다 

### F. Automatic construction and dynamics
* Current KG 는 manual construction에 크게 의존한다 (labor-intensive and expensive)
  * different 인지지능분야에서 KG가 사용되려면 automatic KG construction from large-scale unstructured content가 중요하다 
  *  최근 연구는 semi-automatic construction under supervision of existing KG를 기반으로 하고 있다. 
  *  Multimodality, heterogeneity, and large-scale application을 고려했을 때 automatic construction이 큰 도전
* 또한 현재 static KG에 집중이 되고 있고, temporal scope validity를 예측하거나, temporal information과 entity dynamics을 고려한 연구는 적다. 
  * Dynamic KG together with learning algorithms and capturing dynamics, can address the limitation of traditional knowledge representation and reasoning by considering the temporal nature.



