---
layout: post
title: "[ICLR'17] GCN - Spectral에서 Spatial으로"
description: >
  GCN 논문의 2장 내용인 Graph Spatial Convolution으로의 유도 과정을 이해하기.

related_posts:
  - _posts/study/2022-03-26-gcn-backgrounds.md

image:      
  path:     /assets/img/blog/gcn/cover.png
category:   study
tags:       papers
---


* this unordered seed list will be replaced by the toc
{:toc}

[지난 포스트](gcn-backgrounds)에서 언급했듯이, GCN 논문은 Spectral graph convolution을 Spatial graph convolution으로 근사하여, 비교적 적은 연산으로도 graph convolution을 수행할 수 있음을 보였다. 어떻게 근사하는지를 설명한 2장에 해당하는 내용들을 차근차근 짚어보고자 한다.  

논문에서는 기존 [Spectral graph convloution](#spectral-graph-convolutions)에서, [Chebyshev polynomial을 사용한 근사](#chebyshev-polynomial을-사용한-근사)를 거쳐, [Layer-wise linear model](#layer-wise-linear-model)를 유도한다.

유도 과정 보다 최종 제안방법이 궁금하신 분들은 [layer-wise-propagation-rule](#layer-wise-propagation-rule) 부터 읽으시면 됩니다.
{:.faded}

[논문 링크](https://arxiv.org/abs/1609.02907)

## Definitions
---

수식에서 자주 등장하는 변수들의 정의는 다음과 같다.  
* 그래프 $$ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $$
  * $$ N $$개의 nodes $$ v_i \in \mathcal{V} $$
  * Edge $$ (v_i, v_j) \in \mathcal{E} $$
  * Adjacency matrix $$ A \in \mathbb{R}^{N \times N} $$
  * Degree matrix $$ D_{ii} = \sum_j{A_{ij}} $$
* 그래프 $$ \mathcal{G} $$의 graph Laplacian matrix $$ \Delta = D-A $$
  * Normalized graph Laplacian $$ L = I_N - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} $$
  * $$ L $$의 eigenvector들을 열로 하는 행렬 $$ U $$
  * $$ L $$의 eigenvalue들을 원소로 갖는 대각행렬 $$ \Lambda $$
* 각 node에 대해 scalar값을 갖는 graph signal $$ x \in \mathbb{R}^N $$
  * $$ U^{\top} x $$는 signal $$ x $$의 graph Fourier transform 이다.

## Spectral에서 Spatial으로
---
### Spectral Graph Convolutions
Spectral convolution은 논문의 식 3과 같이 signal $$ x $$와 filter $$ g_{\theta} = diag(\theta) $$의 곱으로 정의된다.  

$$
g \star x = U g_{\theta} U^{\top} x,
\tag{3}
$$

Spectral convolution의 정의
{:.figcaption}

위 식의 $$ g_{\theta} $$는 $$ \theta \in \mathbb{R}^N $$를 원소로 갖는 대각행렬이다.
이 때, $$ g_{\theta} $$는 normalized graph Laplacian matrix의 eigenvalues $$ \Lambda $$를 입력받는 함수인 $$ g_{\theta}(\Lambda) $$로 해석할 수 있다.  

그런데 (3)번 식의 계산 복잡도는 $$ \mathcal{O}(N^2) $$ 으로 상당히 비효율적이다.
뿐만 아니라, 그래프의 크기가 커지면 커질수록 $$ L $$의 고유값 분해는 불가능에 가까울 정도로 계산 복잡도가 높아진다.  

### Chebyshev polynomial을 사용한 근사

계산 복잡도가 높다는 단점을 극복하기 위해, [Hammond et al.](https://arxiv.org/abs/0912.3848)은 Chebyshev polynomials $$ T_k(x) $$를 사용해 filter $$ g_{\theta}(\Lambda) $$를 $$ K $$차원의 다항식으로 근사할 수 있음을 보였다.
학습 해야 하는 파라미터의 개수를 $$ N $$개에서 $$ K $$개로 줄여 계산 복잡도를 낮춘 것이다. Chebyshev polynomial을 사용해 다음 식과 같이 근사된다.

$$
g_{\theta '} (\Lambda) \approx \sum_{k=0}^{K} \theta'_k T_k (\tilde{\Lambda})
\tag{4}
$$

* $$\tilde{\Lambda} = \frac{2}{\lambda_{\max}} \Lambda - I_N $$ 이다. 이 때, $$ \lambda_{\max} $$는 $$L$$의 가장 큰 eigenvalue이다. $$ \tilde{\Lambda} $$는 모든 eigenvalue들이 [-1, 1] 사이의 값을 갖도록 $$ \lambda_{\max} $$로 rescale한 것이다.
* $$ \theta^{'} \in \mathbb{R}^K $$는 Chebyshev coefficient이다.
* Chebyshev polynomial은 $$T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$와 같이 재귀적으로 정의된다. 이 때, $$T_0(x) = 1$$, $$T_1(x) = x$$ 이다.

이 때, $$ U \Lambda^{k} U^{\top} = (U \Lambda U^{\top})^k = L^k $$ 이므로, $$ \Lambda $$의 다항식은 $$ L $$로 표현할 수 있다.
따라서 spectral graph convolution 식은 아래와 같이 표현할 수 있다.

$$
g_{\theta '} \star x \approx \sum_{k=0}^{K} \theta'_k T_k (\tilde{L})x
\tag{5}
$$

* $$ \tilde{L} = \frac{2}{\lambda_{\max}} - I_N $$ 이다.

위 식에서 주목해야 할 점은, convolution 연산에 더이상 $$\Lambda$$가 사용되지 않는다는 점이다. 
결국 **$$L$$의 $$K$$차 다항식으로 spectral graph convolution을 근사**할 수 있다는 것이다.
Graph laplacian matrix $$L$$을 최대 $$K$$번 곱한다는 의미인데, 이는 다시 말해 최대 $$K$$-hop 떨어져 있는 이웃의 차이를 고려해줄 수 있다는 것이다.  

Graph laplacian matrix는 연결된 노드와의 signal 값의 차이가 얼마나 나는지를 연산하는 difference operator이다.
{:.faded}

또한 이렇게 변환했을 때의 계산 복잡도는 $$\mathcal{O}(|\mathcal{E}|)$$로 훨씬 더 간단하게 연산할 수 있다.
뿐만 아니라 더이상 고유값 분해를 하지 않아도 되기 때문에 계산 복잡도 면에서 훨씬 더 효율적이라고 할 수 있다.


### Layer-Wise Linear Model

5번 식을 활성화 함수를 추가하여 여러번 쌓으면 뉴럴 네트워크를 구성할 수 있다. 여기서 저자는 $$K = 1$$로 제한하였다. 논문에서는 이렇게 했을 때의 이점을 다음과 같이 서술한다.

* Explicit하게 Chebyshev polynomial을 사용한 parameter에 제한 되지 않는 convolutional filter를 학습 할 수 있다. (제한을 두지 않기 때문에 더 유용한 정보를 학습할 수 있음.)  
* Local neighborhood 구조에 대한 **오버피팅 문제를 완화**할 수 있다.  
* 각 layer의 연산을 줄여 더 깊은(deep) 뉴럴 네트워크를 구성할 수 있다.  

개인적인 생각으로는, 세번째 이유 때문에 $$K$$를 제한한 것 같다. 뉴럴 네트워크의 한 layer는 통상적으로 linear한 연산 뒤에 non-linear한 활성함수를 수행하여 구성한다. $$K = 1$$로 제한하면 Laplacian matrix의 일차식을 사용하여 linear한 연산을 수행하고, 활성함수를 사용해 통상적인 뉴럴 네트워크와 구조가 유사해진다.
{:.faded}  

더 나아가 $$ \lambda_{\max} \approx 2 $$로 제한을 두어 5번 식을 6번 식을 파라미터 $$\theta_0^{'}$$와 $$\theta_1^{'}$$ 2개 만을 갖는 다음 식으로 간소화 할 수 있다.

$$
g_{\theta '} \star x \approx \theta'_0 x + \theta'_1(L-I_N)x = \theta'_0 x - \theta'_1 D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x
\tag{6}
$$

파라미터의 개수를 제한하면 오버피팅 문제를 줄이고 layer당 연산을 줄일 수 있기 때문에, 한단계 더 나아가 파라미터 1개를 갖도록 $$\theta = \theta_0' = - \theta_1'$$ 이라고 가정하면, 아래와 같이 정리된다.


$$
g_{\theta '} \star x \approx \theta (I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) x
\tag{7}
$$

여기서 $$I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$의 eigenvalue는 [0,2] 사이의 값을 갖기 때문에 exploding/vanishing gradient 문제가 발생할 수 있다. 
이를 방지 하기 위해 논문에서는 *renormalization trick*을 사용하여 [-1, 1]사이의 값을 갖도록 해준다. 
* $$ I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} A \tilde{D}^{-\frac{1}{2}} $$, with $$\tilde{A} = A + I_N$$ and $$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

마지막으로 입력 신호와 filter의 개수를 일반화 해보자. 
모든 노드가 $$C$$차원 벡터 신호를 갖는 상황이면, graph signal은 $$X \in \mathbb{R}^{N \times C}$$ 이다.
또, filter가 $$F$$개 있다고 하면 convolution 연산을 다음 식과 같이 정리할 수 있다.

$$
Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta
\tag{8}
$$

* $$\Theta \in \mathbb{R}^{C \times F}$$: filter의 parameter
* $$Z \in \mathbb{N \times F}$$: convolution 연산의 결과

$$ \tilde{A}X $$는 sparse matrix와 dense matrix의 곱이기 떄문에 edge의 갯수 만큼 연산되어, 
최종적인 계산 복잡도는 $$ \mathcal{O}(|\mathcal{E}|FC) $$이다.


## Layer-wise Propagation Rule
---

본 논문에서 제안한 layer-wise propagation rule은 최종적으로 다음과 같이 정리된다.  

$$
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}).
\tag{2}
$$

$$W^{(l)}$$은 $$l$$번째 layer의 weight matrix이고 $$\sigma(\cdot)$$은 $$\operatorname{ReLU}$$와 같은 활성함수이다. 

위 식을 뒤에서 부터 단계별로 해석해보면 다음과 같다.
1. $$ H^{(l)}W^{(l)} $$: Layer $$(l)$$의 representation과 filter $$W^{(l)}$$의 곱 (filter를 적용하여 필요한 정보만을 추출)
2. $$\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$: 연결되어 있는 노드들의 값을 더해주는 연산
3. $$\sigma(\cdot)$$: 활성함수 적용  


여기서 연결되어 있는 노드들을 합해줄 때, 그냥 $$A$$을 사용해도 되지만, $$\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$을 사용했다. 그 이유가 무엇일까?

첫번째로,  $$ \tilde{A} = A + I_N $$는, *self-connection*을 추가해 새로운 layer의 representation에 이전 자신의 정보를 반영할 수 있게 한 것이다. Self-connection이 없다면 인접 행렬의 대각 요소들은 0값을 갖기 때문에 전 단계의 나의 representation이 반영되지 않는다.

다음으로 $$\tilde{D}^{-\frac{1}{2}}$$를 앞뒤로 곱한 것은 $$\tilde{A}$$의 모든 행의 원소의 합이 1이 되도록 *정규화* 해준 것이다. 정규화 해주지 않으면 이웃노드가 많은 노드와 이웃노드가 적은 노드의 scale이 달라진다. 이는 exploding gradient 등의 문제로 원할한 학습에 방해가 될 수 있다. 따라서 각 노드의 degree로 나누어 정규화를 해준다.


## 결론
---
기존 Spectral graph convolution에서 가장 문제가 되었던 계산 복잡도를 크게 줄여 간단한 연산으로도 graph convolution을 수행할 수 있음을 보였다. 

또한 CNN에서의 2D convolution연산을 graph에 일반화했다. 정보를 뽑아내는 filter가 모든 픽셀/노드에 대해 적용된다. 특히, GCN에서의 convolution은 **지역적인(local) 정보를 취합** 한다는 점에서 기존의 spectral convolution 보다 더 2D convolution과 비슷하다고 할 수 있을 것 같다.  
* 2D convolution은 filter의 convolution 연산을 통해 중심 픽셀의 주변 픽셀들의 정보를 취합
* Graph convolution은 filter matrix의 곱을 통해 중심 노드와 연결되어 있는 노드들의 정보를 취합





## 참고글
---
* [저자 블로그](http://tkipf.github.io/graph-convolutional-networks/)