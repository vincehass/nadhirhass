---
title: Approximate Bayesian Neural Networks
date: 2021-12-31T9:30:00
slug: 2021-12-14-BNN-slides
categories:
  - Probabiistic Machine Learning
  - Approximate Inference
  - Bayesian optimization
  - Hyperparameter optimization
tags:
  - Laplace Approximation
  - Gaussian Processes
  - Numerical Optimization
event: Departement of Mathematics and Industrial Engineering
event_url: https://www.polymtl.ca/renseignements-generaux/en/departments/
location: Montreal, Canada
subtitle: Bayesian pragmatism provides useful tools to tackle overfitting problems, but it comes at a cost, the exact Bayesian inference appropriate to a neural network is often intractable. Bayesian deep learning remains a good choice to design efficient methods by providing an approximate solution combining approximate inference and scalable optimization framework. However, the practical effectiveness of Bayesian neural networks is limited by the need to specify meaningful prior distributions, and by the intractability of posterior inference.
#date_end: 2020-08-31T19:40:00
all_day: no
publishDate: '2021-05-31'
featured: yes
image: 
  caption: '[Bayesian Inference | Hyperparameter optimization| Deep Learning](https://spcanelon.github.io/tour-of-the-tidyverse)'
  focal_point: center
  preview_only: no
slides: ''
projects: []
links:
- icon: images
  icon_pack: fas
  name: slides
  url: 'https://curvertino.netlify.app/pdf/BNN_approx_view-27.pdf'
# - icon: play-circle
#   icon_pack: fas
#   name: video
#   url: https://youtu.be/m_ZoMmAIx-o
# - icon: github
#   icon_pack: fab
#   name: materials
#   url: https://github.com/spcanelon/tour-of-the-tidyverse#an-antarctic-tour-of-the-tidyverse
---

<script src="{{< blogdown/postref >}}index_files/fitvids/fitvids.min.js"></script>


## Description

we address these issues by attempting to demystify the relationship between approximate inference and optimization approaches through the generalized Gauss–Newton method. Bayesian deep learning yields good results, combining Gauss–Newton with Laplace and Gaussian variational approximation. Both methods compute a Gaussian approximation to the posterior; however, it remains unclear how these methods affect the underlying probabilistic model and the posterior approximation. Both methods allow a rigorous analysis of how a particular model fails and the ability to quantify its uncertainty.
<br><br>
