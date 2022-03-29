---
layout: post
title: "[ICLR'17] GCN - Backgrounds"
description: >
  GCN 논문을 읽기 전에 알면 좋은 개념들. Spectral Graph 이론 이해하기.
sitemap:    true
category:   study
tags:       papers
---


* this unordered seed list will be replaced by the toc
{:toc}


GCN 논문을 처음 접했을 때 Graph Laplacian과 Fourier Transform등 처음 보는 용어들이 한꺼번에 등장하여 정말 혼란스러웠다.. 이에 관련해서 찾아본 내용들을 정리해 보았다.   


먼저, GCN에서 하고자 하는 것이 **Spectral graph convolution에서 Spatial graph convolution으로의 근사**이다. 본격적으로 논문을 읽기 전에 전반적인 graph signal과 spectral graph이론에 대해서 알고 있으면 논문이 하고자 하는 방향을 이해하기 쉬울 것이라고 생각한다. 엄밀하게 수학적으로 맞는 설명은 아니지만, 이해한 바를 최대한 쉽게 설명하려고 노력했다..!

논문의 직접적인 내용은 이후에 작성할 예정이다.  



## 1. Graph signal
---

Graph signal은 entity간 연결관계 정보를 반영하는 signal로, $$ N $$개의 노드로 이루어진 graph에서 node마다 scalar 값을 갖는다.  


![Graph signal](/assets/img/blog/2022-03-24/graph-signal.png)
{:.centered}  

Graph signal $$f$$. ([출처](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf))
{:.figcaption}


이와 같은 graph signal을 처리하는데 있어서, **frequency(graph spectral)**측면에서의 접근법이 있는데, graph signal에서의 frequency 개념을 이해하기 위해서는 먼저 graph Laplacian matrix를 알아야 한다.  


## 2. Graph Laplacian matrix
---
### 2.1. Definition of the Graph Laplacian matrix

$$
\begin{aligned}
  \mathcal{G} = &(\mathcal{V}, \mathcal{E})
  \text{, where } v_i \in \mathcal{V}, (v_i, v_j) \in \mathcal{E}
\end{aligned}
$$


위와 같은 그래프 $$\mathcal{G}$$가 주어졌을 때, 해당 그래프의 Laplacian matrix는 아래와 같이 Adjcency matrix $$A \in \mathbb{R}^{N \times N}$$ 에서 Degree matrix $$D_{ii} = \sum_{j}{A_{ij}}$$를 뺀 행렬로 정의된다.  


$$
\begin{aligned}
  L = D - A
\end{aligned}
$$


### 2.2. Laplacian matrix의 의미
한편 graph signal $$ f: \mathcal{V} \rightarrow \mathbb{R}^N $$이 주어졌을 때, $$L$$을 곱해주면 한 node와 연결되어 있는 node들의 signal의 차이를 나타내는 **difference operator**로 작용한다. 


$$
\begin{aligned}
  Lf   &= (D-A)f = Df - Af \\ 
  Lf_i &= \sum_{j=1}^{N} A_{i,j}(f_j - f_j)
\end{aligned}
$$  

Signal에 Laplacian matrix를 곱하는 것은 signal $$f$$의 $$i$$번째 값(노드 $$i$$의 signal 값)에서 다른 노드의 signal 값들을 빼는 연산이다.
{:.figcaption}



### 2.3. Laplacian quadratic form과 graph signal의 frequency
그런데 단순히 graph signal과 Laplacian graph의 곱만을 계산하면 마이너스 값으로 인해 값이 상쇄되므로, 차이 값을 제곱하는 **Laplacian quadratic form**을 사용한다.  


(Quadradic formula는 이차식을 의미한다.)
{:.faded}  


$$
\begin{aligned}
  f^{\top} L f &= \sum
  _i f_i \cdot Lf_i= \sum_{i}\sum_{j} A_{i,j} \space f_i(f_i - f_j)  \\
  &= \frac{1}{2}\sum_{i,j}A_{i,j}\space f_i(f_i-f_j) + \frac{1}{2}\sum_{i,j}A_{j,i}\space f_i(f_i-f_j) \\
  &= \frac{1}{2}\sum_{i,j}A_{i,j}\space f_i(f_i-f_j) - \frac{1}{2}\sum_{i,j}A_{i,j}\space f_j(f_i-f_j) \\
  &= \frac{1}{2} \sum^{N}_{i,j = 1} A_{i,j} \space (f_i-f_j)^2 
\end{aligned}
$$  


이러한 Laplacian quadratic form $$f^{\top}Lf$$은 **graph signal $$f$$가 얼마나 smooth한지** (즉, graph signal이 그래프의 연결성을 잘 반영하는지)를 나타내는 값으로 사용된다.  

다시 말해, graph signal이 smooth하다는 것은 연결되어 있는 node끼리 유사한 signal값을 갖는다는 의미이며, Laplacian quadratic form 값이 작을수록 노드 간 signal 값의 차가 작은 것이므로 해당 signal이 smooth하다고 할 수 있다.  


![frequency](/assets/img/blog/2022-03-24/frequency.png)
{:.centered}  

왼쪽이 오른쪽 보다 smooth한 graph signal이다. ([출처](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf))
{:.figcaption}


이 때, graph signal의 smoothness는 "frequency"라고도 한다. 통상적인 주파수의 의미와 smoothness를 연결해서 생각해 보면, 주파수가 낮은 신호는 주파수가 높은 신호보다 smooth 하다고도 볼 수 있다. 같은 의미에서 smooth한 graph signal은 frequency가 작다고 할 수 있다. 
즉, graph signal의 Laplacian quadratic form 값은 해당 그래프의 frequency라 볼 수 있다.  

결론적으로 다음과 Lapalcian quadratic form과 graph signal의 frequency, smoothness는 다음과 같이 정리할 수 있다.  


> Laplacian quadratic form의 값의 작은 graph signal  
> = frequecy가 작은 graph signal의 frequecy가 작음  
> = smooth한 graph signal  
{:.lead}


Laplacian quadratic form을 다시 한 번 살펴보면 GCN 논문 (1)번 식에 나오는 graph Laplacian regualarization term $$ \mathcal{L}_{reg} $$과 동일한 것을 알 수 있다. 해당 regularization term이 하고자 하는 것은 결국, *node feature $$ X $$를 최대한 smooth한 값으로 변환해주는 함수 $$ f $$를 찾는다*는 의미이다.  
{:.note}


### 2.4. Eigenvalues and eigenvectors of the graph Laplacian matrix

앞서서 graph signal의 frequency에 대해서 살펴 보았다. 다음으로 $$L$$의 eigenvector와 eigenvalue에 대해 살펴보자.  

Eigenvector와 eigenvalue의 정의에 따라, Lapalcian matrix의 eigenvalue와 eigenvector는 식 $$L u_i = {\lambda}_i u_i$$를 만족하는 벡터 $$u_i$$와 스칼라 값 $${\lambda}_i$$이다.  


위 식의 각 변에 $$u_i^{\top}$$을 곱해주면 $$u_i^{\top}L u_i = \lambda_i$$가 되는데, 이는 Laplacian quadratic form 형태이다. 즉, Laplacian matrix의 각 eigenvalue는 각 eigenvector의 frequency로 해석할 수 있는 것이다.  
{:.note}



## 3. Graph Fourier Transform and spectral space
---
Graph Fourier transform은 graph signal $$f$$를 spectral space의 $$\hat{f}$$로 변환하는 것을 말한다.  

이는 graph Laplacian matrix $$ L $$의 eigenvector들의 (weighted)합으로 원래의 graph siganl $$ f $$를 표현하는 것이다. 즉, $$ L $$의 eigenvector들을 기저(basis)로 하는 공간으로의 변환을 의미한다.  

이는 spatial space의 graph signal을 spectral space(i.e., frequency space, 즉 frequency를 의미하는 eigenvalue들이 기준이 되는 공간)으로 변환하는 것이다.  


$$
\hat{f} = U^{\top} f
$$

Spatial space의 벡터 $$ f $$를 spectral space의 벡터 $$ \hat{f} $$로 변환  
{:.figcaption}  


![gft](/assets/img/blog/2022-03-24/gft.png)
{:.centered}

Graph Fourier transform. 그림의 $$\mathcal{X}$$는 eigenvector이다.
{:.figcaption}


## 4. Classical graph spectral filtering
---
일반적인 signal frequency filtering에서는 Fourier transformation을 통해 frequency space에서 해당 signal을 표현하고, 필터(transfer function)를 적용한 뒤, 다시 inverse 연산을 하여 원래 space로 복원한다. 이런 과정을 통해 **signal의 noise를 제거하고 필요한 성분만을 남기게 된다**고 한다.  

이러한 과정을 graph signal에 대해 적용하는 것이 graph spectral filtering이다.  


![graph spectral filtering](/assets/img/blog/2022-03-24/spectral-conv.png)
{:.centered}  

Graph spectral filtering 과정. 그림의 $$\mathcal{X}$$는 eigenvector이다.  
{:.figcaption}  


해당 과정을 단계별로 살펴보면 다음과 같다.  

1. Graph signal을 Graph Fouier Transform(GFT)를 사용해 spectral space로 변환
2. 필터 $$ \hat{g}(\Lambda) $$ 적용  
  * $$
      \hat{g}(\Lambda) = 
        \begin{bmatrix}
        \hat{g}(\lambda_0) & & 0 \\
        & \ddots & \\
        0 & & \hat{g}(\lambda_{N-1})
        \end{bmatrix}
    $$  
  * 이 때 필터 $$ \hat{g}(\Lambda) $$는 spectral space에서 어떤 frequency를 사용할지를 결정한다. 그림에서 filter 3가지가 이에 해당한다. 특정 주파수를 증폭하거나 사용하지 않아도 되는 주파수는 걸러주어 결과적으로 denoise 해주는 역할을 수행한다.
  * 예를 들어 Low-pass filter (그림에서 가장 왼쪽 필터)를 사용하면, low frequency를 갖는 eigen value만을 사용하는 것이다.
3. Inverse-GFT(IGFT)를 사용해 spatial space로 복원

이 3단계를 식으로 표현하면 $$ U \hat{g}(\Lambda) U^{\top} $$이 되는데, 논문의 (3)번 식도 동일한 과정을 거치는 것이다.


## 5. Spectral graph convolution 
---
딥러닝 이전의 spectral filtering은 그림의 3가지 필터와 같은 이미 정의된 필터 $$\hat{g}$$를 사용하였으나, 필터를 parameterize하여 학습하는 것이 바로 Spectral graph convolution이다.

$$
g_{\theta} \star x = U g_{\theta} U^{\top} x \tag{3}
$$

GCN 논문의 spectral graph convolution 식
{:.figcaption}


## 참고글
---

* [GCN(Graph Convolutional Networks)1편: Graph Laplacian부터 Graph Fourier Transform까지 (Spectral Graph Theory)](https://ahjeong.tistory.com/14)
* [MIT Media Lab 자료](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf)
