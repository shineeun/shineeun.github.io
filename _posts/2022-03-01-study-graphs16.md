---
layout: post
title:  "[논문리뷰] TeMP: Temporal Message Passing for Temporal Knowledge Graph Completion "
subtitle:  "Graph"
categories: study
tags: graphs
comments: true
header-img:
---
Wu, J., Cao, M., Cheung, J. C. K., & Hamilton, W. L. (2020). Temp: Temporal message passing for temporal knowledge graph completion. arXiv preprint arXiv:2010.03526.

## Abstract
Inferring missing facts in temporal knowledge graphs (TKGs) is a fundamental and challenging task. Previous works have approached this problem by augmenting methods for static knowledge graphs to leverage time-dependent representations. However, these methods do not explicitly leverage multi-hop structural information and temporal facts from recent time steps to enhance their predictions. Additionally, prior work does not explicitly address the temporal sparsity and variability of entity distributions in TKGs. We propose the Temporal Message Passing (TeMP) framework to address these challenges by combining graph neural networks, temporal dynamics models, data imputation and frequency-based gating techniques. Experiments on standard TKG tasks show that our approach provides substantial gains compared to the previous state of the art, achieving a 10.7% average relative improvement in Hits@10 across three standard benchmarks. Our analysis also reveals important sources of variability both within and across TKG datasets, and we introduce several simple but strong baselines that outperform the prior state of the art in certain settings.

## 1. Introduction
#### Research background
* Temporal knowledge graph(TKGs) 에서 missing fact를 추론하는 능력은 event prediction, QA, social network analysis, recommendation system에서 유용하다. 
* TKGs에서는 기존의 static KGs에서 정보를 (Obama, visit, China)와 같이 triple로 나타내는 것이 아니라, time stamp를 추가하여 (Obama, visit, China, 2014)와 같이 나타낸다. 
* 기존에는 TKGs는 static KG snapshot이 연속으로 구성된 것으로 표현되어짐 
* temporal knowledge graph completion (TKGC): 이러한 snapshot들 간의 사라진 정보를 예측하는 것

#### 선행연구
* TKGC와 관련한 연구들은 time-dependent scoring function을 발전시키는 데 집중
  * missing fact의 likelihood score을 계산
  * static KGs의 representation learning method를 기반으로 함
* **한계** 
  * (1) TKGs의 multi-hop structural information을 제대로 고려하지 않음
  * (2) KG snapshot 근처의 temporal facts를 leverage하는 능력이 없음 
    * (Obama, make agreement with, China, 2013) or (Obama, visit, China, 2012) is useful for answering the query (Obama, visit, ?, 2014)
    * *EWMA와 유사하게 최근 시점의 일에 weight를 더 주는 것인 듯*
  * (3) Temporal variability and temporal sparsity

#### Temporal variability and temporal sparsity
* Temporal variability
  * different queries를 해결하기 위해 near KG snapshot에 reference할 temporal information이 많은 것
    * political event dataset에서 (Obama, visit)에 관련한 quadruples가 (Trump, visit)에 비해 2008-2013년 사이에 더 많을 것이므로, Obama가 2014년에 _어디에_ 방문했는지에 대하 reference한 정보가 많아진다.     
* Temporal sparsity
  * 각 시간 step별로 전체 entity 중 일부만 active한 것
    * 기존 방법론에서는 **inactive entity에도 동일한 embedding을 지정**하여, time-sensitive feature을 잘 표현하지 못하였음

#### Temporal Message Passing (TeMP)
* Neural message passing + temporal dynamic models
* Temporal sparsity와 variability issue 해결 위해 Frequency-based gating + data imputation techniques 사용
* 효과
  * TKGC benchmark에서 SOTA를 달성함
  * Hits@10에서 next best model에 비해 7.3%이상의 성능 향상을 이룸
  * Fine-grained error analysis: variability를 고려하는 것이 중요함

## 2. Related Work
#### Static KG representation learning
* entity와 relation을 low-dimensional embedding으로 표현함
  * entity와 relation embedding기반으로 candidate fact에 대한 score을 계산하는 decoding method를 기반으로 하며, random negative example에 비해 valid triples가 더 높은 score을 가질 수 있도록 모델이 학습이 된다. 
  * 이러한 모델들은 entity와 관련한 multi-hop 정보를 leverage하기 위해 embedding을 생성하기 위해 shallow encoder(ex. single embedding-lookup layers)이나 GNN기반의 message passisng에 의존한다. 

#### Temporal KG representation learning
* 기존 접근법 1
  * Shallow encoder을 통해 embedding을 하고, time-sensitive quadruple decoding function을 적용함. 
    * Time-specific information은 고려되지만, event periodicity와 같은 entity-level에서의 temporal pattern은 고려되지 못함
* 기존 접근법 2
  * Message passing networks를 사용하여 intra-graph의 이웃 정보를 capture
    * temporal recurrence나 attention mechanism과 합쳐진다.
  * 1) Temporal point processes (Han et al., 2020)
    * Continuous TKGC를 파악하는 데 집중
  * 2) methods that resemble TeMP
    * Recurrent Event Networks (RE-NET) (Jin et al., 2019)
      * multi-level RNN을 사용하여 entity interaction을 modeling함 
    * DySAT (Sankar et al., 2020)
      * Self-attention을 사용하여 dynamic graph에 대한 latent node representation을 학습 
    * 이 두가지 방법은 graph extrapolation (sequence에서 next time-step을 예측하는 것)으로 TKGC와는 다른 방법론이다. 

## 3. Proposed Approach
Goal: TKG의 missing fact를 예측하는 것
* TKG *G*={G(1),G(2),...,G(T)}, where G(t)=(E,R,D(t))
* 여기서 E와 R은 모든 시간대에 걸친 entity와 relation의 union set을 의미한다. 
* D(t)는 time t에 존재하는 모든 observed triple의 set을 의미함 

#### Overview of TeMP
Encoder와 decoder의 관점으로 설명
* Encoder
  * 각 entity를 각 time-step t별로 time-dependent low-dimensional embedding z_i,t를 진행
  * structural entity representation과 temporal representaiton을 combine
    * Structural encoder (SE): multi-relational message passing network에 기반으로 하여, entity representation을 진행
      ![image](https://user-images.githubusercontent.com/60350933/156308446-2dc98dce-d652-4121-9e78-e85e9e5464db.png)

    * Temporal encoder (TE): 이전 시간대의 SE의 결과를 통합하여 time-dependent low-dimensional embedding z_i,t를 추론함
      ![image](https://user-images.githubusercontent.com/60350933/156308569-40a484ca-07b0-4067-af20-d473eb314f0d.png)
      * τ: number of temporal input KG snapshots to the model
* Decoder
  * Entity의 embedding을 Temporal fact의 likelihood를 점수로 변환
![image](https://user-images.githubusercontent.com/60350933/156306721-6a227701-523c-450e-9f8a-45fafd97307c.png)

#### 3.1. Structure Encoder
* 각 time-step G(t)의 entity embedding을 생성
* ![image](https://user-images.githubusercontent.com/60350933/156310103-96ac4a4a-a99f-43da-96b7-5391cfae1c07.png)
  * u_i: one-hot embedding indicating entity e_i
  * W0: entity embedding matrix
  * W_r(l), W_s(l)=transformation matrices specific to each layer of the model
    * 모든 time stamp에 있어서 공유됨
  * N_ir: relation r로 연결된 e_i의 set of neighboring entities
    * 이 size는 neighborhood information을 평균화하여, normalizing constant의 역할을 한다. 
   * L개의 layer을 통해서 G(t)의 snapshot에 대해 message-passing 방법론을 사용한 후에, 그 결과로 나오는 entity e_i에 대한 structural embedding entity를 x_i,t로 표현하며, 이는 G(t) 내의 L-hop neighborhood를 요약하는 역할을 한다. 
   ![image](https://user-images.githubusercontent.com/60350933/156320416-c92a2690-c314-429a-ae1c-af92b8ac74f7.png)

* Structural encoder로 RGCN을 사용하지만, CompGCN이나, EdgeGAT와 같은 multi-relational message passing network에 다 적용할 수 있음   

#### 3.2. Temporal Encoder
Entity representation의 across-time information을 통합하는 것
* 방법1: Recurrent architecture 
  * Temporal Recurrence model (TeMP-GRU)
    * weight decay를 traditional recurrence mechanism에 추가하여 historical facts의 diminishing한 효과를 반영하고자 한다. 
    * t-가 entity e_i가 t이전에 active한 상태였던 마지막 시간이라고 하면, down-weighted entity representation은 다음과 같다. 
      ![image](https://user-images.githubusercontent.com/60350933/156324937-3a1f214c-7047-4d68-8328-db108493d629.png)
       *  γz: λz와 bz를 학습가능한 파라미터로 가지는 decay rate이며, 0-1 사이의 값을 가지고, temporal difference에 따라서 monotonically decreasing하는 특성을 가지고 있다. 
       *  ![image](https://user-images.githubusercontent.com/60350933/156325236-b85242b7-e10a-403c-a6bd-cfa81e860344.png)일 때만 down-weighted entity representation이 nonzero가 된다. 
     *  Gated recurrent unit (GRU)를 사용하여 entity embedding z_i,t를 구한다. 
        ![image](https://user-images.githubusercontent.com/60350933/156325420-46e51856-3f12-4e43-ba72-ba67a5b8d7c7.png)
     
* 방법2: Self-attention approach
  * Temporal self attention model (TeMP-SA)
  * Active temporal entity representation의 sequence에 선택적으로 접근함을 통해 historical information을 반영할 수 있다. 
  * Transformer 아키텍쳐를 기반으로하여 각 time-step t'의 entity embedding에 attentive pooling을 적용하여, time-dependent embedding을 적용한다. 
    ![image](https://user-images.githubusercontent.com/60350933/156326599-2aa65149-1fa8-4f43-928a-929ccb050e24.png)
    * Wq, Wk, Wv는 transformer layer에서처럼 linear projection matrix
    * β는 multiplicative attention function을 통해 얻어진 attention weight matrix
    * {λz, bz}는 down-weighting function에서의 학습 파라미터
    * M: mask
      ![image](https://user-images.githubusercontent.com/60350933/156326968-ac1f5861-69cd-411f-b24f-e88d0ae92676.png)
      * M이 마이너스 무한대로 갈 수록 attention weights βij는 0에 수렴함. 이는 **active한 temporal entity representation만이 non-zero weight를 가질 수 있도록 함**
  * Full-self attention model은 multiple attention head를 사용하여 생성되어질 수 있다. 

#### Incorporating future information
* bi-directional GRU를 적용 (recurrent approach)하거나 past와 future time steps를 모두 attend하는 방식(attention-based approach)으로 적용

#### Tackling Temporal Heterogeneities
Data imputation과 frequencey-based gating technique 사용하여 temporal heterogeneities를 다룸
* Dataset마다 heterogeneities는 달라질 수 있으므로, 이러한 heterogeneities를 다루는 것은 옵셔널 하다. 
* Imputation of inactive entities
  * Structural encoder은 동일한 KG snapshot 내 존재하는 neighboring entities를 encode하는 구조이다. 
  * Time step t에서 활성화되지 않은 entity e_i의 경우, static representation x_i,t의 경우 다른 이웃들에 의해서 informed되지 않게되며, multiple time step 간의 stale한 representation이 공유가된다. 
  * Imputation (IM)을 통해서 stale representations에 temporal representation을 통합하게 된다. 
  * 따라서 imputed structural representation은 
  ![image](https://user-images.githubusercontent.com/60350933/156346205-8ecd3216-5296-4c32-95fe-3fc96e1d47ff.png)
  * 이 때 IM은 **weighted sum function**으로 Equation(1)과 유사하게 exponential decay가 적용이 된다. 
    ![image](https://user-images.githubusercontent.com/60350933/156953157-21cf4a82-2bbb-4621-9cf4-0f4c2085fc01.png)
    * exponential decay: 시간에 따라 일정한 비율로 감소하도록 하는 것. 
  * Imputed representation
    ![image](https://user-images.githubusercontent.com/60350933/156954472-7962ce25-1f71-49e8-bd84-6562814ff185.png)
    * 이 때, static model에서 x_i,t를 x_hat_i,t 로 변경하여 사용함을 통해서 temporal model로 활용할 수 있다. 

* Frequency-base gating (FG)
  * entity의 encoded representation의 경우 얼마나 최근 temporal facts와 관련이 있는지에 영향을 많이 받는다. 
  * structural output과 temporal embedding을 융합할 수 있도록 gating term을 적용
  * Query type (subject/object)과 entity position (queried fact에서 entity가 subject인지 object인지)에 따라 weight을 미분하여 quadruple에서의 역할을 문맥화할 수 있게 한다. 
  * **Pattern**
    * non-empty set of the quadruple (s,r,o,t)
  * **Temporal Frequency** of pattern
    * Number of facts with such patterns in the defined time window
    * ex. (Obama, visit, China, 2014)라는 패턴이 존재할 때, (Obama, visit)의 패턴의 frequency는 t'가 timewindow 2000-2014에 속할 때의 (Obama, visit, *,t')의 quadruple의 개수를 의미함. 
  * quadraple (s,r,o,t)의 **Temporal pattern frequencies (TPFs)**
    * (1) subject frequency
      ![image](https://user-images.githubusercontent.com/60350933/156958633-87e0e259-1e18-4e7d-9879-103d146848c5.png)
    * (2) Object frequency ![image](https://user-images.githubusercontent.com/60350933/156975099-e9d75367-1776-4c16-b061-348785a6ef51.png)
    * (3) relation frequency ![image](https://user-images.githubusercontent.com/60350933/156975121-47d841d0-03c4-41ad-b8cc-1d914d174a3c.png)
    * (4) Subject-relation frequency
    * (5) Relation-object frequency
  * Gating mechanism을 quadruple에서 사라진 object를 예측하는 관점에서 정의를 함
    * Subject로 설정을 해도 딤 (Appendix A.3.)
    * Object에서 예측을 하는 것이라면, 모델은 Fs=[subject frequency,relation frequency, subject-relation frequency]에 대해서만 접근이 가능하며, 이 frequency vector Fs에만 대하여 query의 embedding에 대한 gating term을 정의함
      ![image](https://user-images.githubusercontent.com/60350933/156975608-c3697fc9-fe6f-4414-ab4f-8337d21d792e.png)
      ![image](https://user-images.githubusercontent.com/60350933/156975622-0f76a669-ee94-40c6-92dc-96ed30373d80.png)
      * α_o,s와 α_oo는 two-layer dense neural network를 통해서 학습된 [0,1] 사이의 weight이다. 
#### 3.4. Decoder와 학습
* φ(.): tuple에 대한 score
* DEC: static KG에 대한 decoding function (ex. TransE decoder)
* The score for quadruple
  * ![image](https://user-images.githubusercontent.com/60350933/156975837-bb8903d2-fbbd-4f64-a198-ec23c92c4efa.png)
    * ![image](https://user-images.githubusercontent.com/60350933/156975868-7990e4e4-e8dc-4637-b96b-9ffdc43e07ca.png): subject and object embeddings
    * z_r: relation r에 대한 학습된 엠베딩
* 이 score function을 사용하여 모델을 학습한다면, the model parameters are learned using gradient-based optimization in mini-batches
* 각 triple  η =(s, r, o) ∈ D(t)에 대하여, negative set of entities D−_η= {o'|(s, r,o')not in D(t)}을 추출하고 cross-entropy loss를 정의한다.
  ![image](https://user-images.githubusercontent.com/60350933/156976836-ddb5cd6f-f28c-4f8d-a658-83b733cc6c98.png)

* 동일한 과정을 subject에 대해서도 진행을 해준 후, final training loss를 L=L_sub+L_obj로 두어 학습을 한다. 

## 4. Experiments
#### 4.1. Dataset
* Global database of events, language and ton (GDELT)
* Integrated Crisis Early Warning System (ICEWS)- ICEWS14, ICEWS05-15
#### 4.2. Evaluation Metrics
모든 quadraple (s,r,o,t)에 대해서, (s,r,?,t)와 (?,r,o,t)에 대해 평가를 진행한다. 
* (s',r,o,t)에 대해서 score을 계산함
* 각 쿼리에 대해서 (s,r,o,t)의 rank를 계산하여 metrics를 계산함
* Hits@1, @3, @10 scores와 MRR을 통해서 평가를 진행함

#### 4.3. Baseline methods
* Rule-based baseline
  * Temporal exponential decay
    * rule-based baseline
    * Copying facts from quadruples in the recent past and future
    * 가까운 시간의 facts를 복사하면서 missing fact를 계산함
    * copying fact의 확률은 (1) queried quadruple과 겹치는 elements의 number을 계산하거나 (2) 현재 timestep의 temporal distance와 가까운 fact에 기반함  
* Static embedding methods: input에 time을 고려하지 않음
  * TransE
  * DistMult
  * ComplEx
  * SimplE
  * Static RGCN (SRGCN): RCGN messaging 방법론을 활용
* SOTA TKGC methods
  * TTransE
  * TADistMult
  * HyTe
  * Diachronic Embedding
  * AtisEE
  * AtisER
  * TNTComplEx 
#### 4.4. Implementation and hyperparameters
* PyTorch lightning, Deep Graph Library

#### 4.5. Results and analysis
* TeMP-GRU와 TeMP-SA를 사용하고, frequency-based gating과 imputation은 옵셔널로 둠
![image](https://user-images.githubusercontent.com/60350933/156978488-a40b859e-e2ac-487a-8918-b460ae6199cf.png)

* TeMP-GRU와 TeMP-SA 모두 SOTA를 달성함 (Hits@10 관점에서)
* Baseline 모델 중에 rule-based TED baseline이 모든 데이터셋에서 강력한 성능을 보임
  * queried quadruples와 동일한 패턴의 temporal facts를 커피하는 것의 강력함
  * SRGCN의 경우에도 높은 성능을 보였는데 이는 message-passing 절차가 모델로 하여금 multi-hop structural info를 time-step별로 잘 leverage할 수 있기 때문임
* IM과 FG에 대한 test
  * ![image](https://user-images.githubusercontent.com/60350933/156978563-4908e5e1-366a-4e69-9f86-df9f6127500a.png)
  * 모든 timestemp에서 모든 entity가 active하다면 IM과 FG의 성능은 효과가 없다. 

#### 4.5.3 Fine-grained Error Analysis
* TKGC 쿼리가 temporal pattern frequency가 달라짐에 따라 어떻게 모델의 성능이 달라지는 지 평가하기 위해 quadruples를 다른 TPF로 그룹지어 각 그룹별로 Hits@10을 계산
* subject-relation frequency를 subject와 object query에 대한 모델 성능과 비교하여 *replication*과 *reference* 효과를 파악
  * Replication effect: Temporal fact로부터 동일한 정답을 복사하여 예측하는 것
    * ex. (Obama, visit,?,2014)를 (Biden, visit, China, 2013)으로부터 복사하여 예측하는 것
  * Reference 효과: 유사한 fact를 활용하여 temporal context를 통해서 답을 포함하는 것
    * ex. Obama가 2013년에 방문한 국가들 중에서 2014년에 간 곳을 유추하는 것
* Static 모델과 비교
  * replication 모델의 경우 TeMP-GRU 모델의 성능이 더 좋음
    * TeMP 모델들이 TKGC에 있어서 temporal facts를 사용하는데 더 유리하다는 것을 알 수 있으며, 모든 TFPs에 있어서 frequency-based gating이 전반적인 성능 향상에 도움을 준다. 
  * reference 효과에 있어서는 TPFs가 크면 클수록 성능이 더 떨어진다.
    * Temporal model이 object query performance와 subject-relation TPF간의 non-linear correlation이 존재함을 알 수 있다. 
    * 즉, Obama가 2008년부터 2013년까지 100개 이상의 국가를 방문했다면 China를 선택하는 것이 더 어려울 것이다. 
    * 따라서 gating의 경우, high TPFs를 가지는 query에 있어서 중요한 역할을 한다. 

## 5. Conclusion
In this work, we present a novel framework named TeMP for temporal knowledge graph completion (TKGC). TeMP computes entity representation by jointly modelling multi-hop structural information and temporal facts from nearby time-steps.
Additionally, we introduce novel frequencybased gating and data imputation techniques to address the temporal variability and sparsity problems in TKGC. We show that our model is able to achieve superior performance (10.7% relative improvement) over the state-of-the-arts on three benchmark datasets. 
Our work is potentially beneficial to other tasks such as temporal information extraction and temporal question answering, by providing beliefs about the likelihood of facts at particular points in time.
Future work involves exploring the generalization of TeMP to continuous TKGC and better imputation techniques to induce representations for infrequent and inactive entities. 

## References
Busbridge, D., Sherburn, D., Cavallo, P., & Hammerla, N. Y. (2019). Relational graph attention networks. arXiv preprint arXiv:1904.05811.
Jin, W., Qu, M., Jin, X., & Ren, X. (2019). Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. arXiv preprint arXiv:1904.05530.
Han, Z., Ma, Y., Wang, Y., Günnemann, S., & Tresp, V. (2020). Graph hawkes neural network for forecasting on temporal knowledge graphs. arXiv preprint arXiv:2003.13432.
Sankar, A., Wu, Y., Gou, L., Zhang, W., & Yang, H. (2020, January). Dysat: Deep neural representation learning on dynamic graphs via self-attention networks. In Proceedings of the 13th International Conference on Web Search and Data Mining (pp. 519-527).
Vashishth, S., Sanyal, S., Nitin, V., & Talukdar, P. (2019). Composition-based multi-relational graph convolutional networks. arXiv preprint arXiv:1911.03082.

