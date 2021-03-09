**This repository contains all of the code that we used for our Seminar Case Study 2020/2021.**

This repository consists of several .py files:
* DataAnalysis.py: This script contains several functions that can tell us more about the data we are provided with (e.g. value ranges)
* MainData.py: This script processes the data (handling NA values, adding new variables and creating balanced datasets)
* DataImbalance.py: This script contains our K-Means undersampling method for making a dataset balanced 
* MainModels.py: Running this script generates forecasts for the four criteria based on XGBoost and Neural Network models
* NeuralNetwork.py: This script contains the function for computing forecasts with a Neural Network model
* XGBoost.py: This script contains the function for computing forecasts with an XGBoost model
* PredictionMetrics.py: This script contains several functions that can be called in MainModels.py to evaluate the performance of the model predictions 

*As we are not allowed to share any of the data we work with, we use a gitignore file to prevent all local csv files from being pushed into this public repository.* 
*For all code that contains a random component, we set the seed equal to 1234.*

*Authors: Lucas van den Adel, Anne Jasmijn Langerak, Thijs Schrijvers, Tim van den Toorn*
