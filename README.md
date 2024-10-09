# [Model Deployment : Estimating Heart Failure Survival Risk Profiles From Cardiovascular, Hematologic And Metabolic Markers](https://johnpaulinepineda.github.io/Portfolio_Project_55/)

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#) [![](https://img.shields.io/badge/Github-black?logo=Github)](#) [![](https://img.shields.io/badge/Streamlit-black?logo=Streamlit)](#)

This [project](https://johnpaulinepineda.github.io/Portfolio_Project_55/) aims to develop a web application to enable the accessible and efficient use of a survival prediction model for estimating the heart failure survival probability and predicting the risk category of a test case, given various cardiovascular, hematologic and metabolic markers. The model development process implemented the Cox Proportional Hazards Regression, Cox Net Survival, Survival Tree, Random Survival Forest, and Gradient Boosted Survival models as independent base learners to estimate the survival probabilities of right-censored survival time and status responses, while evaluating for optimal hyperparameter combinations (using Repeated K-Fold Cross Validation), imposing constraints on model coefficient updates (using Ridge and Elastic Net Regularization, as applicable), and delivering accurate predictions when applied to new unseen data (using model performance evaluation with Harrel's Concordance Index on Independent Validation and Test Sets). Additionally, survival probability functions were estimated for model risk-groups and the individual test case. Creating the prototype required cloning the repository containing two application codes and uploading to Streamlit Community Cloud - a Model Prediction Code to estimate heart failure survival probabilities, and predict risk categories; and a User Interface Code to process the study population data as baseline, gather the user input as test case, render all user selections into the visualization charts, execute all computations, estimations and predictions, indicate the test case prediction into the survival probability plot, and display the prediction results summary. The final heart failure survival prediction model was deployed as a [Streamlit Web Application](https://heart-failure-survival-probability-estimation.streamlit.app).

<img src="images/ModelDeployment2_Summary_0.png?raw=true"/>

<img src="images/ModelDeployment2_Summary_1.png?raw=true"/>

<img src="images/ModelDeployment2_Summary_2.png?raw=true"/>

<img src="images/ModelDeployment2_Summary_3.png?raw=true"/>
