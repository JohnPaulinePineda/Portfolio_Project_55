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
from model_prediction import compute_individual_coxph_survival_probability_class, compute_list_coxph_survival_profile, bin_numeric_model_feature, plot_kaplan_meier

##################################
# Defining file paths
##################################
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_PREPROCESSED_PATH = r"datasets\preprocessed"
PIPELINES_PATH = r"pipelines"

##################################
# Loading the dataset
# from the DATASETS_FINAL_TRAIN_PATH
# and the DATASETS_PREPROCESSED_PATH
##################################
X_train = pd.read_csv(os.path.join("..",DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"), index_col=0)
y_train = pd.read_csv(os.path.join("..",DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"), index_col=0)
x_original_EDA = pd.read_csv(os.path.join("..",DATASETS_PREPROCESSED_PATH, "heart_failure_EDA.csv"), index_col=0)

##################################
# Rebuilding the training data
# for Kaplan-Meier plotting
##################################
X_train_indices = X_train.index.tolist()
x_original_MI = x_original_EDA.copy()
x_original_MI = x_original_MI.drop(['DIABETES','SEX', 'SMOKING','CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
x_train_MI = x_original_MI.loc[X_train_indices]

##################################
# Converting the event and duration variables
# for the train set to array 
# as preparation for modeling
##################################
y_train_array = np.array([(row.DEATH_EVENT, row.TIME) for index, row in y_train.iterrows()], dtype=[('DEATH_EVENT', 'bool'), ('TIME', 'int')])

##################################
# Binning the numeric features
# into dichotomous categories
##################################
for numeric_column in ["AGE","EJECTION_FRACTION","SERUM_CREATININE","SERUM_SODIUM"]:
    x_train_MI_EDA = bin_numeric_model_feature(x_train_MI, numeric_column)

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
    serum_creatinine_numeric_input = st.slider(variables[2], min_value=0.5, max_value=10.0, value=0.5)
    serum_sodium_numeric_input = st.slider(variables[3], min_value=110, max_value=150, value=50)
    anaemia_categorical_input = st.radio(variables[4], ('Present', 'Absent'), horizontal=True)
    high_blood_pressure_categorical_input = st.radio(variables[5], ('Present', 'Absent'), horizontal=True)
    anaemia_numeric_input = 1 if anaemia_categorical_input == 'Present' else 0
    high_blood_pressure_numeric_input = 1 if high_blood_pressure_categorical_input == 'Present' else 0

test_case_responses[variables[0]] = age_numeric_input
test_case_responses[variables[1]] = ejection_fraction_numeric_input
test_case_responses[variables[2]] = serum_creatinine_numeric_input
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

##################################
# Defining the code logic
# for the button action
##################################    
if entered:
    ##################################
    # Defining a section title
    # for the test case characteristics
    ##################################    
    st.markdown("""---""")      
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Characteristics</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 

    ##################################
    # Applying preprocessing to 
    # the test case
    ##################################
    coxph_pipeline = joblib.load(os.path.join("..",PIPELINES_PATH, "coxph_pipeline.pkl"))
    coxph_pipeline.fit(X_train, y_train_array)
    X_test_sample_transformed = coxph_pipeline.named_steps['yeo_johnson'].transform(X_test_sample)
    X_test_sample_converted = pd.DataFrame([X_test_sample_transformed[0]], columns=["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM", "ANAEMIA", "HIGH_BLOOD_PRESSURE"])

    ##################################
    # Binning numeric predictors into two groups
    ##################################
    X_test_sample_converted = bin_numeric_model_feature(X_test_sample_converted, ["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM"])

    ##################################
    # Converting integer predictors into labels
    ##################################
    for col in ["ANAEMIA", "HIGH_BLOOD_PRESSURE"]:
        X_test_sample_converted[col] = X_test_sample_converted[col].apply(lambda x: 'Absent' if x < 1.0 else 'Present')

    ##################################
    # Creating a 2x3 grid of plots for
    # plotting the estimated survival profiles
    # of the test case input
    # using Kaplan-Meier Plots
    # against the training data characteristics
    ##################################
    fig, axes = plt.subplots(3, 2, figsize=(17, 13))

    heart_failure_predictors = ['AGE','EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM','ANAEMIA','HIGH_BLOOD_PRESSURE']

    for i, predictor in enumerate(heart_failure_predictors):
        ax = axes[i // 2, i % 2]
        plot_kaplan_meier(x_train_MI_EDA, predictor, ax, new_case_value=X_test_sample_converted[predictor][0])
        ax.set_title(f'Baseline Survival Probability by {predictor} Categories')
        ax.set_xlabel('TIME')
        ax.set_ylabel('Estimated Survival Probability')
        ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)

    ##################################
    # Defining a section title
    # for the test case heart failure survival probability estimation
    ################################## 
    st.markdown("""---""")    
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Heart Failure Survival Probability Estimation</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 

    ##################################
    # Computing the estimated survival probabilities
    # for the test case at five defined time points
    # and determining the risk category 
    ##################################
    X_test_sample_prediction = compute_individual_coxph_survival_probability_class(X_test_sample)
    X_test_sample_survival_function = X_test_sample_prediction[0]
    X_test_sample_prediction_50 = X_test_sample_prediction[1]
    X_test_sample_prediction_100 = X_test_sample_prediction[2]
    X_test_sample_prediction_150 = X_test_sample_prediction[3]
    X_test_sample_prediction_200 = X_test_sample_prediction[4]
    X_test_sample_prediction_250 = X_test_sample_prediction[5]
    X_test_sample_risk_category = X_test_sample_prediction[6]
    X_test_sample_survival_time = X_test_sample_prediction[7]
    X_test_sample_survival_probability = X_test_sample_prediction[8]

    ##################################
    # Plotting the estimated survival probability
    # for the test case 
    # in the baseline survival function
    # of the final survival prediction model
    ##################################
    X_train_survival_function = compute_list_coxph_survival_profile(X_train)

    ##################################
    # Resetting the index for 
    # plotting survival functions
    # for the training data
    ##################################
    y_train_reset_index = y_train.reset_index()

    ##################################
    # Creating a 1x1 plot
    # for plotting the estimated survival probability profile
    # of the final survival prediction model
    ##################################
    fig, ax = plt.subplots(figsize=(17, 8))

    for i, surv_func in enumerate(X_train_survival_function):
        ax.step(surv_func.x, 
                surv_func.y, 
                where="post", 
                color='red' if y_train_reset_index['DEATH_EVENT'][i] == 1 else 'blue', 
                linewidth=6.0,
                alpha=0.05)
    if X_test_sample_risk_category == "Low-Risk":
        ax.step(X_test_sample_survival_function[0].x, 
                X_test_sample_survival_function[0].y, 
                where="post", 
                color='blue',
                linewidth=6.0,
                linestyle='-',
                alpha=0.30,
                label='Test Case (Low-Risk)')
        ax.step(X_test_sample_survival_function[0].x, 
                X_test_sample_survival_function[0].y, 
                where="post", 
                color='black',
                linewidth=3.0,
                linestyle=':',
                label='Test Case (Low-Risk)')
        for survival_time, survival_probability in zip(X_test_sample_survival_time, X_test_sample_survival_probability):
            ax.vlines(x=survival_time, ymin=0, ymax=survival_probability, color='blue', linestyle='-', linewidth=2.0, alpha=0.30)
        red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
        blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
        black_patch = plt.Line2D([0], [0], color='black', lw=3, linestyle=":", label='Test Case (Low-Risk)')
    if X_test_sample_risk_category == "High-Risk":
        ax.step(X_test_sample_survival_function[0].x, 
                X_test_sample_survival_function[0].y, 
                where="post", 
                color='red',
                linewidth=6.0,
                linestyle='-',
                alpha=0.30,
                label='Test Case (High-Risk)')
        ax.step(X_test_sample_survival_function[0].x, 
                X_test_sample_survival_function[0].y, 
                where="post", 
                color='black',
                linewidth=3.0,
                linestyle=':',
                label='Test Case (High-Risk)')
        for survival_time, survival_probability in zip(X_test_sample_survival_time, X_test_sample_survival_probability):
            ax.vlines(x=survival_time, ymin=0, ymax=survival_probability, color='red', linestyle='-', linewidth=2.0, alpha=0.30)
        red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
        blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
        black_patch = plt.Line2D([0], [0], color='black', lw=3, linestyle=":", label='Test Case (High-Risk)')
    ax.legend(handles=[red_patch, blue_patch, black_patch], facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
    ax.set_title('Final Survival Prediction Model: Cox Proportional Hazards Regression')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Estimated Survival Probability')
    plt.tight_layout(rect=[0, 0, 1.00, 0.95])
    st.pyplot(fig)

    ##################################
    # Defining a section title
    # for the test case model prediction summary
    ##################################      
    st.markdown("""---""")   
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Model Prediction Summary</h4>", unsafe_allow_html=True)    
    st.markdown("""---""")

    ##################################
    # Summarizing the test case model prediction results
    ##################################     
    if X_test_sample_risk_category == "Low-Risk":
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (50 Days): <span style='color:blue;'>{X_test_sample_prediction_50:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (100 Days): <span style='color:blue;'>{X_test_sample_prediction_100:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (150 Days): <span style='color:blue;'>{X_test_sample_prediction_150:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (200 Days): <span style='color:blue;'>{X_test_sample_prediction_200:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (250 Days): <span style='color:blue;'>{X_test_sample_prediction_250:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Predicted Risk Category: <span style='color:blue;'>{X_test_sample_risk_category}</span></h4>", unsafe_allow_html=True)
    
    if X_test_sample_risk_category == "High-Risk":
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (50 Days): <span style='color:red;'>{X_test_sample_prediction_50:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (100 Days): <span style='color:red;'>{X_test_sample_prediction_100:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (150 Days): <span style='color:red;'>{X_test_sample_prediction_150:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (200 Days): <span style='color:red;'>{X_test_sample_prediction_200:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability (250 Days): <span style='color:red;'>{X_test_sample_prediction_250:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Predicted Risk Category: <span style='color:red;'>{X_test_sample_risk_category}</span></h4>", unsafe_allow_html=True)
    
    st.markdown("""---""")



