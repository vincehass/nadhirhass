---
title: "GFlowOut: Dropout with Generative Flow Networks"
author: "Dianbo Liu, Moksh Jain, Bonaventure Dossou, Qianli Shen, Salem Lahlou, Anirudh Goyal, Nikolay Malkin, Chris Emezue, Dinghuai Zhang, Nadhir Hassen, Xu Ji, Kenji Kawaguchi, Yoshua Bengio"
date: '2022-11-07'
slug: GFN_project
categories:
  - Research
  - Journal of Machine Learning
tags:
  - Hyper-parameter optimization
  - Bayesian Optimization
  - Machine Learning
  - Monte Carlo
  - Optimization
  - Uncertainty quantification
doi: ''
publishDate: '2022-11-07T16:26:27-05:00'
publication_types:
  - '2'
publication: "In *Machine Learning Research*"
publication_short: 'Arxiv2022'
abstract: 'Bayesian Inference offers principled tools to tackle many critical problems with modern neural networks such as poor calibration and generalization, and data inefficiency. However, scaling Bayesian inference to large architectures is challenging and requires restrictive approximations. Monte Carlo Dropout has been widely used as a relatively cheap way for approximate Inference and to estimate uncertainty with deep neural networks. Traditionally, the dropout mask is sampled independently from a fixed distribution. Recent works show that the dropout mask can be viewed as a latent variable, which can be inferred with variational inference. These methods face two important challenges: (a) the posterior distribution over masks can be highly multi-modal which can be difficult to approximate with standard variational inference and (b) it is not trivial to fully utilize sample-dependent information and correlation among dropout masks to improve posterior estimation. In this work, we propose GFlowOut to address these issues. GFlowOut leverages the recently proposed probabilistic framework of Generative Flow Networks (GFlowNets) to learn the posterior distribution over dropout masks. We empirically demonstrate that GFlowOut results in predictive distributions that generalize better to out-of-distribution data, and provide uncertainty estimates which lead to better performance in downstream tasks.'

summary: 'A key shortcoming of modern deep neural networks is that
they are often overconfident about their predictions, especially when there is a distributional shift between train and
test dataset Daxberger et al. (2021); Nguyen et al. (2015);
Guo et al. (2017). In risk-sensitive scenarios such as clinical practice and drug discovery, where mistakes can be extremely costly, it is important that models provide predictions with reliable uncertainty estimates Bhatt et al. (2021). Bayesian Inference offers principled tools to model the parameters of neural networks as random variables, placing a
prior on them and inferring their posterior given some observed data MacKay (1992); Neal (2012). T'
featured: yes
links:
  - icon: file-pdf
    icon_pack: fas
    name: Arxiv
    url: 'https://arxiv.org/abs/2210.12928'
  - icon: doi
    icon_pack: ai
    name: Machine Learning 2022
    url: 'https://arxiv.org/pdf/2210.12928.pdf'
  # - icon: file-pdf
  #   icon_pack: fas
  #   name: pdf
  #   url_source: '/BOReport.pdf'
# - icon: github
#   icon_pack: fab
#   name: Repo
#   url: https://github.com/CodeForPhilly/datahack2020/tree/master/analyses/team11
# - icon: github
#   icon_pack: fab
#   name: Orion Project
#   url: https://orion.readthedocs.io/en/stable/
# - icon: images
#   icon_pack: fas
#   name: Slides
#   url: pdf/OrionDaskSummit.pdf
image:
  caption: ''
  placement: 2
  focal_point: 'Center'
  preview_only: no
projects: []
slides: ''
---

## Abstract

Bayesian Inference offers principled tools to tackle many critical problems with modern neural networks such as poor calibration and generalization, and data inefficiency. However, scaling Bayesian inference to large architectures is challenging and requires restrictive approximations. Monte Carlo Dropout has been widely used as a relatively cheap way for approximate Inference and to estimate uncertainty with deep neural networks. Traditionally, the dropout mask is sampled independently from a fixed distribution. Recent works show that the dropout mask can be viewed as a latent variable, which can be inferred with variational inference. These methods face two important challenges: (a) the posterior distribution over masks can be highly multi-modal which can be difficult to approximate with standard variational inference and (b) it is not trivial to fully utilize sample-dependent information and correlation among dropout masks to improve posterior estimation. In this work, we propose GFlowOut to address these issues. GFlowOut leverages the recently proposed probabilistic framework of Generative Flow Networks (GFlowNets) to learn the posterior distribution over dropout masks. We empirically demonstrate that GFlowOut results in predictive distributions that generalize better to out-of-distribution data, and provide uncertainty estimates which lead to better performance in downstream tasks.

## Generative Flow Networks.



![poster](GFOCapture.PNG)

