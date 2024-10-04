##################################
# Loading Python Libraries
##################################
import os
import joblib
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter

##################################
# Defining file paths
##################################
MODELS_PATH = r"models"
PARAMETERS_PATH = r"parameters"

##################################
# Loading the final survival prediction model
# from the MODELS_PATH
##################################
final_survival_prediction_model = joblib.load(os.path.join("..",MODELS_PATH, "coxph_best_model.pkl"))

##################################
# Loading the final survival prediction model
# risk group threshold
# from the PARAMETERS_PATH
##################################
final_survival_prediction_model_risk_group_threshold = joblib.load(os.path.join("..",PARAMETERS_PATH, "coxph_best_model_risk_group_threshold.pkl"))

##################################
# Formulating a function to
# generate the heart failure survival profile,
# estimate the heart failure survival probabilities,
# and predict the risk category
# of an individual test case
##################################
def compute_individual_coxph_survival_probability_class(X_test_sample):
    X_test_sample_survival_function = final_survival_prediction_model.predict_survival_function(X_test_sample)
    X_test_sample_risk_category = "High-Risk" if (final_survival_prediction_model.predict(X_test_sample) > final_survival_prediction_model_risk_group_threshold) else "Low-Risk"
    X_test_sample_survival_time = np.array([50, 100, 150, 200, 250])
    X_test_sample_survival_probability = np.interp(X_test_sample_survival_time, 
                                                   X_test_sample_survival_function[0].x, 
                                                   X_test_sample_survival_function[0].y)
    X_test_sample_prediction_50, X_test_sample_prediction_100, X_test_sample_prediction_150, X_test_sample_prediction_200, X_test_sample_prediction_250  = X_test_sample_survival_probability*100
    return X_test_sample_prediction_50, X_test_sample_prediction_100, X_test_sample_prediction_150, X_test_sample_prediction_200, X_test_sample_prediction_250, X_test_sample_risk_category

##################################
# Formulating a function to
# generate the heart failure survival profile and
# estimate the heart failure survival probabilities
# of a list of train cases
##################################
def compute_list_coxph_survival_profile(X_train_list, y_train_list):
    X_list_survival_function = final_survival_prediction_model.predict_survival_function(X_train_list)
    y_list_response = y_train_list.reset_index()
    return X_list_survival_function, y_list_response
    