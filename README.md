# ODEs-informed Machine Learning for Epidemic Spreading Forecasting

## Description
1. EDA/: 
* ARIMA.ipynb: Synthetic arima time series generation.
* LV_Data.ipynb: Synthetic Lotka–Volterra dynamics generation.
* S(E)IR_Simulation.ipynb: Synthetic S(E)IR dynamics generation.
* SuEIRD_Simulation.ipynb: Synthetic SEIRD dynamics generation.
* EDA_Covid19.ipynb: Explotary data analysis of COVID-19 trajectories.
* Statistical_Analysis_Covid19.ipynb: Statistical a nalysis of COVID-19 trajectories.
   
2. Learn_Single_ODE_Sample/: 
* FC-Generalization.ipynb: Experiments on testing the generalization ability of dense neural nets.
* Learn_Single_Sample_LV.ipynb: Learn the single LV sample with ODE and DL methods.
* Learn_Single_Sample_SEIR.ipynb: Learn the single LV sample with ODE and DL methods.
* neural_odes.py: Neural LV module.

3. Main/:
* Evaluation.ipynb: contains the functions of four evaluation metrics.
* Run_model.ipynb: a helper function for calculating energy spectrum.
* SuEIRD_Piecewise.ipynb: Pytorch implementation of AdjMask_SuEIRD. 
* ode_nn: Pytorch implementation of Seq2Seq, Auto-FC, Transformer, Neural ODE, AutoODE, Graph Attention, Graph Convolution.


## Requirement
* python 3.6
* pytorch 10.1
* matplotlib
* scipy
* numpy
* pandas
* dgl
