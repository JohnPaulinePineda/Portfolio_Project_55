##################################
# Loading Python Libraries
##################################
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter
from model_prediction import compute_individual_coxph_survival_probability_class, compute_list_coxph_survival_profile, bin_numeric_model_feature

##################################
# Defining file paths
##################################
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_PREPROCESSED_PATH = r"datasets\preprocessed"

##################################
# Loading the dataset
# from the DATASETS_FINAL_TRAIN_PATH
##################################
X_train = pd.read_csv(os.path.join("..",DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join("..",DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"))
x_original_EDA = pd.read_csv(os.path.join("..",DATASETS_PREPROCESSED_PATH, "heart_failure_EDA.csv"))

##################################
# Rebuilding the training data
# for plotting kaplan-meier charts
##################################
X_train_indices = X_train.index.tolist()
x_original_MI = x_original_EDA.copy()
x_original_MI = x_original_MI.drop(['DIABETES','SEX', 'SMOKING', 'CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
x_original_MI = x_original_MI.loc[X_train_indices]

##################################
# Binning the numeric features
# into dichotomous categories
##################################
for numeric_column in ["AGE","EJECTION_FRACTION","SERUM_CREATININE","SERUM_SODIUM"]:
    x_original_MI_EDA = bin_numeric_model_feature(x_original_MI, numeric_column)

##################################
# Setting the page layout to wide
##################################
st.set_page_config(layout="wide")

##################################
# Listing the variables
##################################
variables = ['AGE',
             'EJECTION_FRACTION',
             'SERUM_CREATININE',
             'SERUM_SODIUM',
             'ANAEMIA',
             'HIGH_BLOOD_PRESSURE']

##################################
# Initializing lists to store user responses
##################################
test_case_responses = {}

##################################
# Creating a title for the application
##################################
st.markdown("""---""")
st.markdown("<h1 style='text-align: center;'>Heart Failure Survival Probability Estimator</h1>", unsafe_allow_html=True)

##################################
# Providing a description for the application
##################################
st.markdown("""---""")
st.markdown("<h5 style='font-size: 20px;'>This model evaluates the heart failure survival risk of a test case based on certain cardiovascular, hematologic and metabolic markers. Pass the appropriate details below to visually assess your characteristics against the study population, plot your survival probability profile, estimate your heart failure survival probabilities at different time points, and determine your risk category. For more information on the complete model development process, you may refer to this <a href='https://johnpaulinepineda.github.io/Portfolio_Project_55/' style='font-weight: bold;'>Jupyter Notebook</a>. Additionally, all associated datasets and code files can be accessed from this <a href='https://github.com/JohnPaulinePineda/Portfolio_Project_55' style='font-weight: bold;'>GitHub Project Repository</a>.</h5>", unsafe_allow_html=True)

##################################
# Creating a section for 
# selecting the options
# for the test case characteristics
##################################
st.markdown("""---""")
st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Cardiovascular, Hematologic and Metabolic Markers</h4>", unsafe_allow_html=True)
st.markdown("""---""")

##################################
# Creating sliders for numeric features
# and radio buttons for categorical features
# and storing the user inputs
##################################
input_column_1,input_column_2, input_column_3 = st.columns(3)
with input_column_2:
    age_numeric_input = st.slider(variables[0], min_value=20, max_value=100, value=20)
    ejection_fraction_numeric_input = st.slider(variables[1], min_value=10, max_value=80, value=10)
    serum_creatinine_input = st.slider(variables[2], min_value=0.5, max_value=10.0, value=0.5)
    serum_sodium_numeric_input = st.slider(variables[3], min_value=110, max_value=150, value=50)
    anaemia_categorical_input = st.radio(variables[4], ('Present', 'Absent'), horizontal=True)
    high_blood_pressure_categorical_input = st.radio(variables[5], ('Present', 'Absent'), horizontal=True)
    anaemia_numeric_input = 1 if anaemia_categorical_input == 'Present' else 0
    high_blood_pressure_numeric_input = 1 if high_blood_pressure_categorical_input == 'Present' else 0

test_case_responses[variables[0]] = age_numeric_input
test_case_responses[variables[1]] = ejection_fraction_numeric_input
test_case_responses[variables[2]] = serum_creatinine_input
test_case_responses[variables[3]] = serum_sodium_numeric_input
test_case_responses[variables[4]] = anaemia_numeric_input
test_case_responses[variables[5]] = high_blood_pressure_numeric_input

st.markdown("""---""")

##################################
# Converting the user inputs
# to a dataframe
##################################  
X_test_sample = pd.DataFrame([test_case_responses])

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

entered = st.button("Assess Characteristics Against Study Population + Plot Survival Probability Profile + Estimate Heart Failure Survival Probability + Predict Risk Category")





