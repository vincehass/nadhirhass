---
title: "Approximate Bayesian Optimisation for Neural Networks"
author: "Nadhir Hassen & Irina Rish"
date: '2021-07-01'
slug: BO_project
categories:
  - Research
  - Journal of Machine Learning
tags:
  - Hyper-parameter optimization
  - Bayesian Optimization
  - Machine Learning
  - Automating ML
  - Optimization
  - Gaussian Processes
doi: ''
publishDate: '2021-07-10T16:26:27-05:00'
publication_types:
  - '2'
publication: "In *Journal of Machine Learning Research*"
publication_short: 'JMLR2020'
abstract: 'A body of work has been done to automate machine learning algorithm to highlight the importance of model choice. Automating the process of choosing the best forecasting model and its corresponding parameters can result to improve a wide range of real-world applications. Bayesian optimisation (BO) uses a blackbox optimisation methods to propose solutions according to an exploration-exploitation trade-off criterion through acquisition functions. BO framework imposes two key ingredients: a probabilistic surrogate model that consist of prior belief of the unknown objective function(data-dependant) and an objective function that describes how optimal is the model-fit. Choosing the best model and its associated hyperparameters can be very expensive, mostly due to budget constraints or a-non-existing closed form of the surrogate model. In fact, most real-dataset are non-stationary which make idealistic assumptions on surrogate models. The necessity to solve the analytical tractability and the computational feasibility in a stochastic fashion enables to ensure the efficiency and the applicability of Bayesian optimisation. In this paper our goal is provide a link between density-ratio estimation and class probability estimation, this reformulation provides algorithm efficiency and tractability.'

summary: 'A novel Bayesian Optimization method based on a linearized link-function to accounts the under-presented class by using a GP surrogate model. This method is based on Laplace’s method and Gauss-Newton approximations to the Hessian. Our method can improve generalization and be useful when validation data is unavailable (e.g., in nonstationary settings) to solve heteroscedastic behaviours. Our experiments demonstrate that our BO by Gauss-Newton approach competes favorably with state-of-the-art blackbox optimization algorithms.'
featured: yes
links:
  - icon: file-pdf
    icon_pack: fas
    name: Arxiv
    url: 'https://arxiv.org/abs/2108.12461'
  - icon: doi
    icon_pack: ai
    name: Artificial Inteligence Workshop 2022
    url: 'https://curvertino.netlify.app/pdf/Approximate_Bayesian_Optimisation_for_Neural_Networks_accepted.pdf'
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

A body of work has been done to automate machine learning algorithms and to highlight the importance of model choice. Automating the process of choosing the best forecasting model and its corresponding parameters can result to improve a wide range of real-world applications. Bayesian optimisation (BO) uses a black-box optimisation methods to propose solutions according to an exploration-exploitation trade-off criterion through acquisition functions. BO framework imposes two key ingredients: a probabilistic surrogate model that consists of prior belief of the unknown dynamic of the model and an objective function that describes how optimal the model-fit. Choosing the best model and its associated hyperparameters can be very expensive, and is typically fit using Gaussian processes (GPs). However, since GPs scale cubically with the number of observations, it has been challenging to handle objectives whose optimisation requires many evaluations. In addition, most real-datasets are non-stationary which makes idealistic assumptions on surrogate models. The necessity to solve the analytical tractability and the computational feasibility in a stochastic fashion enables to ensure the efficiency and the applicability of Bayesian optimisation. In this paper we explore the use of approximate inference with Bayesian Neural Networks as an alternative to GPs to model distributions over functions. Our contribution is to provide a link between density-ratio estimation and class probability estimation based on approximate inference, this reformulation provides algorithm efficiency and tractability.  


## Automate Machine Learning

Improve Hyperparameter Learning with Approximate Bayesian Optimization.

![[2021 PSB Poster (letter)](PSB2021_poster_letter.pdf)](branin_regret_blog.png)
