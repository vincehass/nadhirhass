---
title: 'Kronecker-factored approximation (KFAC) of the Laplace-GGN for Continual Learning'
date: '2021-05-01'
slug: Continual Learning
categories:
  - Research
tags:
  - Continual Learning
  - Bayesian Deep Learning
  - Optimization
  - Approximate Inference
  - Gaussian Processes
author: "Nadhir Hassen"
#doi: https://doi.org/10.1371/journal.pone.0166392
publishDate: '2021-05-12T01:04:26-04:00'
publication_types:
  - '1'
publication: 'ICML'
publication_short: 'ICML'
abstract: 'One major challenge in Continual Learning is that the model forgets how to solve earlier tasks commonly known as Catastrophic Forgetting. Extensive work has been recently made in this direction relies of weight regularisation but may not be effective for complex deep neural network. Recently functional regularisation was developed to directly regularise the network output, although computationally expensive, is expected to perform better. However this technique limits the posterior expressiveness of fully-factorized Gaussian assumption Cremer et al. (2018) of the inference model. Functional regularisation (Pan et al., 2021) has been developed to combine parameter and function space regularisation but does not take into account structured parameter spaces where have complicated inverse Fisher-matrix computations, which give poor predictives estimates. The method applies a full covariance Gaussian posterior approximation and in case of diagonal posterior approximation it fails to model meaningful relationships. In this work we address this issue without alternating the complexity cost. We propose a Laplace Gauss-Newton approximation to combine local parametrization and function space learning using Kronecker-factored approximation (KFAC) of the Laplace-Gauss-Newton (GGN). We use KFAC (Martens & Grosse, 2020a) posterior approximation for image classification to illustrate how we can model parameter covariances per layer without altering scalability in a Continual Learning setting.'
summary: 'Publication highlighting how Catastrophic Forgetting can be solved by Laplace Gauss-Newton approximation without alternating the complexity cost.'
featured: yes
url_pdf: 'https://curvertino.netlify.app/pdf/KFAC_ICML_vc.pdf'
url_code: ~
url_dataset: ~
url_poster: ~
url_project: ~
url_slides: ~
#url_source: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166392
url_video: ~
image:
  caption: '[Continual Learning| Gaussian Processes|Approximate Inference](https://paperswithcode.com/task/continual-learning)'
  focal_point: ''
  preview_only: no
projects: []
slides: ''
links:
  - icon: file-pdf
    icon_pack: fas
    name: PDF
    url: 'https://curvertino.netlify.app/pdf/KFAC_ICML_vc.pdf'
  # - icon: doi
  #   icon_pack: ai
  #   name: Publication
  #   url: https://doi.org/10.1016/j.ijmedinf.2020.104339
  # - icon: file-pdf
  #   icon_pack: fas
  #   name: AMIA 2020 Poster
  #   url: AMIA2020_poster.pdf
  # - icon: file-pdf
  #   icon_pack: fas
  #   name: Preprint
  #   url_pdf: 'https://curvertino.netlify.app/pdf/KFAC_paper.pdf'
  # - icon: external-link-alt
  #   icon_pack: fas
  #   name: medRxiv Preprint
  #   url: https://www.medrxiv.org/content/10.1101/2020.07.30.20165381v1
---

## Abstract

One major challenge in Continual Learning is that the model forgets how to solve earlier tasks commonly known as Catastrophic Forgetting. Extensive work has been recently made in this direction relies of weight regularisation but may not be effective for complex deep neural network. Recently functional regularisation was developed to directly regularise the network output, although computationally expensive, is expected to perform better. However this technique limits the posterior expressiveness of fully-factorized Gaussian assumption Cremer et al. (2018) of the inference model. Functional regularisation (Pan et al., 2021) has been developed to combine parameter and function space regularisation but does not take into account structured parameter spaces where have complicated inverse Fisher-matrix computations, which give poor predictives estimates. The method applies a full covariance Gaussian posterior approximation and in case of diagonal posterior approximation it fails to model meaningful relationships. In this work we address this issue without alternating the complexity cost. We propose a Laplace Gauss-Newton approximation to combine local parametrization and function space learning using Kronecker-factored approximation (KFAC) of the Laplace-Gauss-Newton (GGN). We use KFAC (Martens & Grosse, 2020a) posterior approximation for image classification to illustrate how we can model parameter covariances per layer without altering scalability in a Continual Learning setting.
