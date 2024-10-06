##################################
# Loading Python Libraries
##################################
import os
import joblib
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter
from pathlib import Path

##################################
# Defining file paths
##################################
MODELS_PATH = Path("models")
PARAMETERS_PATH = Path("parameters")

##################################
# Loading the final survival prediction model
# from the MODELS_PATH
##################################
final_survival_prediction_model = joblib.load(MODELS_PATH / "coxph_best_model.pkl")

##################################
# Loading the numeric feature median
# from the PARAMETERS_PATH
##################################
numeric_feature_median = joblib.load(PARAMETERS_PATH / "numeric_feature_median_list.pkl")

##################################
# Loading the final survival prediction model
# risk group threshold
# from the PARAMETERS_PATH
##################################
final_survival_prediction_model_risk_group_threshold = joblib.load(PARAMETERS_PATH / "coxph_best_model_risk_group_threshold.pkl")

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
    X_test_50, X_test_100, X_test_150, X_test_200, X_test_250  = X_test_sample_survival_probability*100
    return X_test_sample_survival_function, X_test_50, X_test_100, X_test_150, X_test_200, X_test_250, X_test_sample_risk_category, X_test_sample_survival_time, X_test_sample_survival_probability 

##################################
# Formulating a function to
# generate the heart failure survival profile and
# estimate the heart failure survival probabilities
# of a list of train cases
##################################
def compute_list_coxph_survival_profile(X_train_list):
    X_list_survival_function = final_survival_prediction_model.predict_survival_function(X_train_list)
    return X_list_survival_function

##################################
# Formulating a function to
# create dichotomous bins
# for the numeric features
# of a list of train cases
##################################
def bin_numeric_model_feature(X_original_list, numeric_feature):
    median = numeric_feature_median.loc[numeric_feature]
    X_original_list[numeric_feature] = np.where(X_original_list[numeric_feature] <= median, "Low", "High")
    return X_original_list

##################################
# Formulating a function to plot the
# estimated survival profiles
# using Kaplan-Meier Plots
##################################
def plot_kaplan_meier(df, cat_var, ax, new_case_value=None):
    kmf = KaplanMeierFitter()

    # Defining the color scheme for each category
    if cat_var in ['AGE', 'EJECTION_FRACTION', 'SERUM_CREATININE', 'SERUM_SODIUM']:
        categories = ['Low', 'High']
        colors = {'Low': 'blue', 'High': 'red'}
    else:
        categories = ['Absent', 'Present']
        colors = {'Absent': 'blue', 'Present': 'red'}

    # Plotting each category with a partly red or blue transparent line depending on the risk category
    for value in categories:
        mask = df[cat_var] == value
        kmf.fit(df['TIME'][mask], event_observed=df['DEATH_EVENT'][mask], label=f'{cat_var}={value} (Baseline Distribution)')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[str(value)], linestyle='-', linewidth=6.0, alpha=0.30)

    # Overlaying a black broken line for the new case if provided
    if new_case_value is not None:
        mask_new_case = df[cat_var] == new_case_value
        kmf.fit(df['TIME'][mask_new_case], event_observed=df['DEATH_EVENT'][mask_new_case], label=f'{cat_var}={new_case_value} (Test Case)')
        kmf.plot_survival_function(ax=ax, ci_show=False, color='black', linestyle=':', linewidth=3.0)




    