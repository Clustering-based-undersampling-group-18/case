This repository contains all of the code that we used for our seminar case study 2020/2021 research.
Important for replication is to note that for all the code that utilizes random choice, we set the random set equal to 1234.

This repository consists of several .py files:
* DataAnalysis.py: This script contains several functions that can tell us more about the data we are provided with (e.g. value ranges)
* MainData.py: This script processes the data (handling NA values, adding new variables and creating balanced datasets)
* DataImbalance.py: This script contains our K-Means undersampling method for making a dataset balanced 
* MainModels.py: Running this script generates forecasts for the four criteria based on XGBoost and Neural Network models
* NeuralNetwork.py: This script contains the function for computing forecasts with a Neural Network model
* XGBoost.py: This script contains the function for computing forecasts with an XGBoost model
* PredictionMetrics.py: This script contains several functions that can be called in MainModels.py to evaluate the performance of the model predictions 

*As we are not allowed to share any of the data we work with, we use a gitignore file to prevent all local csv files from being pushed into this public repository.* 

*Authors: Lucas van den Adel, Anne Jasmijn Langerak, Thijs Schrijvers, Tim van den Toorn*

