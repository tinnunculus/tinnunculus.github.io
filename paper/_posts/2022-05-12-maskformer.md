---
layout: post
title: MaskFormer
sitemap: false
---

**참고**  
[1] <https://arxiv.org/abs/2107.06278>  
[2] <https://github.com/facebookresearch/MaskFormer>
* * *  

**코드**  
<https://github.com/tinnunculus/SwinTransformer/blob/main/swin.ipynb>    
* * *  

* toc
{:toc}

## Introduction
* MaskFormer는 Object dectection 모델인 DETR을 segmentation task에 맞게 수정한 모델이라고 볼 수 있다.
* 그렇기 때문에 핵심 개념인 모델의 구조와 학습 방법으로는 DETR 모델과 거진 유사하다.
* 기존에는 semantic, instance, panoptic 등의 여러가지 segmentation 문제마다 다르게 접근하여 문제를 풀었지만, 이 논문에서는 하나의 학습된 MaskFormer 모델로 inference 방법만 task 마다 다르게 하여 앞에 언급한 segmentation 문제를 모두 풀 수 있다.

## Mask classification formulation
> <p align="center"><img src="/assets/img/paper/maskformer/1.png"></p>
* DETR 모델에서는 object detection 모델이기에 트랜스포머의 Query 토큰들이 가리키는 것은 object box 정보였다면 MaskFormer에서는 segmentation 정보의 embedding vector를 가리킨다.
* 이 embedding vector는 후에 per-pixel embedding tensor와 곱셈 연산을 통해 mask segmentation 정보를 가리키고 linear mapping을 통해 mask의 class 정보를 기리키게 된다.
* 나머지 개념은 DETR과 동일하다. class는 no object를 포함하고 있고, bipartite maching을 통해 prediction 정보와 ground truth 정보를 1대1 mapping 하고 학습을 진행한다.
* hungarian matching의 score 함수로 $$ -p_i(c^gt_j)