## Paper: 
Rui Wang, Danielle Maddix, Christos Faloutsos, Yuyang Wang, Rose Yu [Bridging Physics-based and Data-driven modeling for
Learning Dynamical Systems](https://arxiv.org/pdf/2011.10616.pdf), Annual Conference on Learning for Dynamics and Control (L4DC), 2021

## Abstract:
How can we learn a dynamical system to make forecasts, when some variables are unobserved? For instance, in COVID-19, we want to forecast the number of infected and death cases but we do not know the count of susceptible and exposed people. While mechanics compartment models are widely-used in epidemic modeling, data-driven models are emerging for disease forecasting. As a case study, we compare these two types of models for COVID-19 forecasting and notice that physics-based models significantly outperform deep learning models. We present a hybrid approach, AutoODE-COVID, which combines a novel compartmental model with automatic differentiation. Our method obtains a 57.4% reduction in mean absolute errors for 7-day ahead COVID-19 forecasting compared with the best deep learning competitor. To understand the inferior performance of deep learning, we investigate the generalization problem in forecasting. Through systematic experiments, we found that deep learning models fail to forecast under shifted distributions either in the data domain or the parameter domain. This calls attention to rethink generalization especially for learning dynamical systems.

## Description
1. ode_nn/: 
* DNN.py: Pytorch implementation of Seq2Seq, Auto-FC, Transformer, Neural ODE.
* Graph.py: Pytorch implementation of Graph Attention, Graph Convolution.
* AutoODE.py: Pytorch implementation of AutoODE(-COVID).
* train.py: data loaders, train epoch, validation epoch, test epoch functions.

3. Run_DSL.ipynb: run model function.
4. Run_AutoODE.ipynb: Pytorch implementation of AdjMask_SuEIRD. 
5. Evaluation.ipynb: evaluation functions and prediction visualization


## Requirement
* python 3.6
* pytorch 10.1
* matplotlib
* scipy
* numpy
* pandas
* dgl


## Cite
```
@inproceedings{wang2021incorporating,
title={Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems},
author={Rui Wang and Danielle Maddix and Christos Faloutsos and Yuyang Wang and Rose Yu},
journal={arXiv preprint arXiv:2011.10616},
year={2020}
}
```
