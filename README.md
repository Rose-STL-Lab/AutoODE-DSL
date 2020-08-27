# ODEs-and-DL

## Description
1. EDA/: 
* ARIMA.ipynb: Synthetic arima time series generation.
* LV_Data.ipynb: Synthetic Lotka–Volterra dynamics generation.
* S(E)IR_Simulation.ipynb: Synthetic S(E)IR dynamics generation.
* SuEIRD_Simulation.ipynb: Synthetic SEIRD dynamics generation.
* EDA_Covid19.ipynb: Explotary data analysis of COVID-19 trajectories.
* Statistical_Analysis_Covid19.ipynb: Statistical a nalysis of COVID-19 trajectories.
   
2. Learn_Single_ODE_Sample/: 
   1. model.py: TF-net pytorch implementation.
   2. penalty.py: a few regularizers we have tried.
   3. train.py: data loaders, train epoch, validation epoch, test epoch functions.
   4. run_model.py: Scripts to train TF-Net

3. Main/:
   1. Evaluation.ipynb: contains the functions of four evaluation metrics.
   2. radialProfile.py: a helper function for calculating energy spectrum.


## Requirement
* python 3.6
* pytorch 10.1
* matplotlib
* scipy
* numpy
* pandas
* dgl
