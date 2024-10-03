***
# Model Deployment : Estimating Heart Failure Survival Risk Profiles From Cardiovascular, Hematologic And Metabolic Markers

***
### John Pauline Pineda <br> <br> *October 5, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Pre-Modelling Data Preparation](#1.6.1)
        * [1.6.2 Data Splitting](#1.6.2)
        * [1.6.3 Modelling Pipeline Development](#1.6.3)
            * [1.6.3.1 Cox Proportional Hazards Regression](#1.6.3)
            * [1.6.3.2 Cox Net Survival](#1.6.3)
            * [1.6.3.3 Survival Tree](#1.6.3)
            * [1.6.3.4 Random Survival Forest](#1.6.3)
            * [1.6.3.5 Gradient Boosted Survival](#1.6.3)    
        * [1.6.4 Cox Proportional Hazards Regression Model Fitting | Hyperparameter Tuning | Validation](#1.6.4)
        * [1.6.5 Cox Net Survival Model Fitting | Hyperparameter Tuning | Validation](#1.6.5)
        * [1.6.6 Survival Tree Model Fitting | Hyperparameter Tuning | Validation](#1.6.6)
        * [1.6.7 Random Survival Forest Model Fitting | Hyperparameter Tuning | Validation](#1.6.7)
        * [1.6.8 Gradient Boosted Survival Model Fitting | Hyperparameter Tuning | Validation](#1.6.9)
        * [1.6.9 Model Selection](#1.6.9)
        * [1.6.10 Model Testing](#1.6.10)
        * [1.6.11 Model Inference](#1.6.11)
    * [1.7 Predictive Model Deployment Using Streamlit and Streamlit Community Cloud](#1.7)
        * [1.7.1 Model Prediction Application Code Development](#1.7.1)
        * [1.7.2 Model Application Programming Interface Code Development](#1.7.2)
        * [1.7.3 User Interface Application Code Development](#1.7.3)
        * [1.7.4 Web Application](#1.7.4)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) relates the time until an event occurs (such as death or disease progression) to one or more predictor variables. The model is expressed through its hazard function, which represents the risk of the event happening at a particular time for an individual, given that the individual has survived up to that time. The mathematical equation is represented by the baseline hazard function (referring to the hazard for an individual when all of their covariates are zero, representing the inherent risk of the event happening over time, but is not directly estimated in the Cox model. Instead, the focus is on how the covariates influence the hazard relative to this baseline) and an exponential term that modifies the baseline hazard based on the individual's covariates (Each covariate is associated with a regression coefficient which measures the strength and direction of the effect of the covariate on the hazard. The exponential function ensures that the hazard is always positive, as hazard values can’t be negative). The proportional hazards assumption in this model means that the ratio of hazards between any two individuals is constant over time and is determined by the differences in their covariates. The Cox model doesn’t require a specific form for the baseline hazard, making it flexible, while properly accounting for censored data, which is common in survival studies.

[Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) are a class of survival analysis models used to analyze time-to-event data by directly modelling the survival time itself. An AFT model assumes that the effect of covariates accelerates or decelerates the life time of an event by some constant factor. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a random variable with a specified distribution. In an AFT model, the coefficients represent the multiplicative effect on the survival time. An exponentiated regression coefficient greater than one prolongs survival time, while a value less than one shortens it. The scale parameter determines the spread or variability of survival times. AFT models assume that the effect of covariates on survival time is multiplicative and that the survival times can be transformed to follow a specific distribution.

[Regularization Methods](http://appliedpredictivemodeling.com/), in the context of binary classification using Logistic Regression, are primarily used to prevent overfitting and improve the model's generalization to new data. Overfitting occurs when a model is too complex and learns not only the underlying pattern in the data but also the noise. This leads to poor performance on unseen data. Regularization introduces a penalty for large coefficients in the model, which helps in controlling the model complexity. In Logistic Regression, this is done by adding a regularization term to the loss function, which penalizes large values of the coefficients. This forces the model to keep the coefficients small, thereby reducing the likelihood of overfitting. Addiitonally, by penalizing the complexity of the model through the regularization term, regularization methods also help the model generalize better to unseen data. This is because the model is less likely to overfit the training data and more likely to capture the true underlying pattern.

[Shapley Additive Explanations](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) are based on Shapley values developed in the cooperative game theory. The process involves explaining a prediction by assuming that each explanatory variable for an instance is a player in a game where the prediction is the payout. The game is the prediction task for a single instance of the data set. The gain is the actual prediction for this instance minus the average prediction for all instances. The players are the explanatory variable values of the instance that collaborate to receive the gain (predict a certain value). The determined value is the average marginal contribution of an explanatory variable across all possible coalitions.

[FastAPI](https://fastapi.tiangolo.com/) is a modern, fast (high-performance) web framework for building APIs with Python. It’s designed to make it easy to create APIs quickly, while still providing strong validation and type hints that improve both code quality and performance. FastAPI allows building APIs by defining endpoints, which are essentially routes that handle HTTP requests. Routes are defined in Python functions, and FastAPI takes care of handling the request, validating input data, and generating responses. Significant features include high performance (FastAPI is one of the fastest Python frameworks, comparable to Node.js and Go, making it ideal for production use, especially in applications that require low latency, such as machine learning model inference), asynchronous support (FastAPI natively supports asynchronous programming, which is great for handling multiple requests concurrently, improving performance in scenarios with high traffic), data validation (FastAPI automatically validates the data coming into the API based on type annotations. If the user sends incorrect data (e.g., wrong data type), FastAPI generates informative error messages), and auto-generated documentation (FastAPI automatically generates API documentation in both Swagger UI and ReDoc formats. This makes testing and understanding the created API simple and fast for developers and stakeholders). In the context of machine learning model deployment, FastAPI acts as the backend API that handles requests for predictions. When a client or application (like a frontend UI) sends data to the API, FastAPI passes it to the model, retrieves the model’s prediction, and sends the result back to the client.

[Streamlit](https://streamlit.io/) is an open-source Python library that simplifies the creation and deployment of web applications for machine learning and data science projects. It allows developers and data scientists to turn Python scripts into interactive web apps quickly without requiring extensive web development knowledge. Streamlit seamlessly integrates with popular Python libraries such as Pandas, Matplotlib, Plotly, and TensorFlow, allowing one to leverage existing data processing and visualization tools within the application. Streamlit apps can be easily deployed on various platforms, including Streamlit Community Cloud, Heroku, or any cloud service that supports Python web applications.

[Streamlit Community Cloud](https://streamlit.io/cloud), formerly known as Streamlit Sharing, is a free cloud-based platform provided by Streamlit that allows users to easily deploy and share Streamlit apps online. It is particularly popular among data scientists, machine learning engineers, and developers for quickly showcasing projects, creating interactive demos, and sharing data-driven applications with a wider audience without needing to manage server infrastructure. Significant features include free hosting (Streamlit Community Cloud provides free hosting for Streamlit apps, making it accessible for users who want to share their work without incurring hosting costs), easy deployment (users can connect their GitHub repository to Streamlit Community Cloud, and the app is automatically deployed from the repository), continuous deployment (if the code in the connected GitHub repository is updated, the app is automatically redeployed with the latest changes), 
sharing capabilities (once deployed, apps can be shared with others via a simple URL, making it easy for collaborators, stakeholders, or the general public to access and interact with the app), built-in authentication (users can restrict access to their apps using GitHub-based authentication, allowing control over who can view and interact with the app), and community support (the platform is supported by a community of users and developers who share knowledge, templates, and best practices for building and deploying Streamlit apps).

## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Heart Failure Dataset](https://paperswithcode.com/dataset/survival-analysis-of-heart-failure-patients) from [Papers With Code](https://paperswithcode.com/) (with all credits attributed to [Saurav Mishra](https://paperswithcode.com/search?q=author%3ASaurav+Mishra)) was used for the analysis as consolidated from the following primary source: 
1. Research Paper entitled **A Comparative Study for Time-to-Event Analysis and Survival Prediction for Heart Failure Condition using Machine Learning Techniques** from the [Journal of Electronics, Electromedical Engineering, and Medical Informatics](http://jeeemi.org/index.php/jeeemi/article/view/225/94)
2. Research Paper entitled **Machine Learning Can Predict Survival of Patients with Heart Failure from Serum Creatinine and Ejection Fraction Alone** from the [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5) Journal

This study hypothesized that cardiovascular, hematologic, and metabolic markers influence heart failure survival risks between patients.

The event status and survival duration variables for the study are:
* <span style="color: #FF0000">DEATH_EVENT</span> - Status of the patient within the follow-up period (0, censored | 1, death)
* <span style="color: #FF0000">TIME</span> - Follow-up period (Days)

The predictor variables for the study are:
* <span style="color: #FF0000">AGE</span> - Patient's age (Years)
* <span style="color: #FF0000">ANAEMIA</span> - Hematologic marker for the indication of anaemia (decrease of red blood cells or hemoglobin level in the blood) (0, Absent | 1 Present)
* <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span> - Metabolic marker for the level of the CPK enzyme in the blood (mcg/L)
* <span style="color: #FF0000">DIABETES</span> - Metabolic marker for the indication of diabetes (0, Absent | 1 Present)
* <span style="color: #FF0000">EJECTION_FRACTION</span> - Cardiovascular marker for the ejection fraction (percentage of blood leaving the heart at each contraction) (%)
* <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span> - Cardiovascular marker for the indication of hypertension (0, Absent | 1 Present)
* <span style="color: #FF0000">PLATELETS</span> - Hematologic marker for the platelets in the blood (kiloplatelets/mL)
* <span style="color: #FF0000">SERUM_CREATININE</span> - Metabolic marker for the level of creatinine in the blood (mg/dL)
* <span style="color: #FF0000">SERUM_SODIUM</span> - Metabolic marker for the level of sodium in the blood (mEq/L)
* <span style="color: #FF0000">SEX</span> - Patient's sex (0, Female | 1, Male)
* <span style="color: #FF0000">SMOKING</span> - Cardiovascular marker for the indication of smoking (0, Absent | 1 Present)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **299 rows** (observations)
    * **13 columns** (variables)
        * **2/13 event | duration** (object | numeric)
             * <span style="color: #FF0000">DEATH_EVENT</span>
             * <span style="color: #FF0000">TIME</span>
        * **6/13 predictor** (numeric)
             * <span style="color: #FF0000">AGE</span>
             * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span>
             * <span style="color: #FF0000">EJECTION_FRACTION</span>
             * <span style="color: #FF0000">PLATELETS</span>
             * <span style="color: #FF0000">SERUM_CREATININE</span>
             * <span style="color: #FF0000">SERUM_SODIUM</span>
        * **5/13 predictor** (object)
             * <span style="color: #FF0000">ANAEMIA </span>
             * <span style="color: #FF0000">DIABETES</span>
             * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
             * <span style="color: #FF0000">SEX</span>
             * <span style="color: #FF0000">SMOKING</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools
import joblib
%matplotlib inline

from operator import add,mul,truediv
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import pointbiserialr

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator

import shap

import warnings
warnings.filterwarnings('ignore')
```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
DATASETS_PREPROCESSED_PATH = r"datasets\preprocessed"
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_FINAL_VALIDATION_PATH = r"datasets\final\validation"
DATASETS_FINAL_VALIDATION_FEATURES_PATH = r"datasets\final\validation\features"
DATASETS_FINAL_VALIDATION_TARGET_PATH = r"datasets\final\validation\target"
DATASETS_FINAL_TEST_PATH = r"datasets\final\test"
DATASETS_FINAL_TEST_FEATURES_PATH = r"datasets\final\test\features"
DATASETS_FINAL_TEST_TARGET_PATH = r"datasets\final\test\target"
MODELS_PATH = r"models"
```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
heart_failure = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "heart_failure_clinical_records_dataset.csv"))
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(heart_failure.shape)
```

    Dataset Dimensions: 
    


    (299, 13)



```python
##################################
# Verifying the column names
##################################
print('Column Names: ')
display(heart_failure.columns)
```

    Column Names: 
    


    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')



```python
##################################
# Removing trailing white spaces
# in column names
##################################
heart_failure.columns = [x.strip() for x in heart_failure.columns]
```


```python
##################################
# Standardizing the column names
##################################
heart_failure.columns = ['AGE', 
                         'ANAEMIA', 
                         'CREATININE_PHOSPHOKINASE', 
                         'DIABETES', 
                         'EJECTION_FRACTION',
                         'HIGH_BLOOD_PRESSURE', 
                         'PLATELETS', 
                         'SERUM_CREATININE', 
                         'SERUM_SODIUM', 
                         'SEX',
                         'SMOKING', 
                         'TIME', 
                         'DEATH_EVENT']
```


```python
##################################
# Verifying the corrected column names
##################################
print('Column Names: ')
display(heart_failure.columns)
```

    Column Names: 
    


    Index(['AGE', 'ANAEMIA', 'CREATININE_PHOSPHOKINASE', 'DIABETES',
           'EJECTION_FRACTION', 'HIGH_BLOOD_PRESSURE', 'PLATELETS',
           'SERUM_CREATININE', 'SERUM_SODIUM', 'SEX', 'SMOKING', 'TIME',
           'DEATH_EVENT'],
          dtype='object')



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(heart_failure.dtypes)
```

    Column Names and Data Types:
    


    AGE                         float64
    ANAEMIA                       int64
    CREATININE_PHOSPHOKINASE      int64
    DIABETES                      int64
    EJECTION_FRACTION             int64
    HIGH_BLOOD_PRESSURE           int64
    PLATELETS                   float64
    SERUM_CREATININE            float64
    SERUM_SODIUM                  int64
    SEX                           int64
    SMOKING                       int64
    TIME                          int64
    DEATH_EVENT                   int64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
heart_failure.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>DIABETES</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>TIME</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Setting certain integer variables
# to float values
##################################
float_columns = ['AGE',
                 'CREATININE_PHOSPHOKINASE',
                 'EJECTION_FRACTION',
                 'PLATELETS',
                 'SERUM_CREATININE',
                 'SERUM_SODIUM',
                 'TIME']
heart_failure[float_columns] = heart_failure[float_columns].astype(float)
```


```python
##################################
# Setting certain integer variables
# to object or categorical values
##################################
int_columns = ['ANAEMIA',
               'DIABETES', 
               'HIGH_BLOOD_PRESSURE',
               'SMOKING',
               'SEX']
heart_failure[int_columns] = heart_failure[int_columns].astype(object)
heart_failure['DEATH_EVENT'] = heart_failure['DEATH_EVENT'].astype('category')
```


```python
##################################
# Saving a copy of the original dataset
##################################
heart_failure_original = heart_failure.copy()
```


```python
##################################
# Setting the levels of the dichotomous categorical variables
# to boolean values
##################################
heart_failure['DEATH_EVENT'] = heart_failure['DEATH_EVENT'].cat.set_categories([0, 1], ordered=True)
heart_failure['SEX'] = heart_failure['SEX'].replace({0: 'Female', 1: 'Male'})
heart_failure[int_columns] = heart_failure[int_columns].replace({0: 'Absent', 1: 'Present'})
```


```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(heart_failure.dtypes)
```

    Column Names and Data Types:
    


    AGE                          float64
    ANAEMIA                       object
    CREATININE_PHOSPHOKINASE     float64
    DIABETES                      object
    EJECTION_FRACTION            float64
    HIGH_BLOOD_PRESSURE           object
    PLATELETS                    float64
    SERUM_CREATININE             float64
    SERUM_SODIUM                 float64
    SEX                           object
    SMOKING                       object
    TIME                         float64
    DEATH_EVENT                 category
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
heart_failure.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>DIABETES</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>TIME</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>Absent</td>
      <td>582.0</td>
      <td>Absent</td>
      <td>20.0</td>
      <td>Present</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130.0</td>
      <td>Male</td>
      <td>Absent</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>Absent</td>
      <td>7861.0</td>
      <td>Absent</td>
      <td>38.0</td>
      <td>Absent</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136.0</td>
      <td>Male</td>
      <td>Absent</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>Absent</td>
      <td>146.0</td>
      <td>Absent</td>
      <td>20.0</td>
      <td>Absent</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129.0</td>
      <td>Male</td>
      <td>Present</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>Present</td>
      <td>111.0</td>
      <td>Absent</td>
      <td>20.0</td>
      <td>Absent</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137.0</td>
      <td>Male</td>
      <td>Absent</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>Present</td>
      <td>160.0</td>
      <td>Present</td>
      <td>20.0</td>
      <td>Absent</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116.0</td>
      <td>Female</td>
      <td>Absent</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration 
# of the numeric variables
##################################
print('Numeric Variable Summary:')
display(heart_failure.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AGE</th>
      <td>299.0</td>
      <td>60.833893</td>
      <td>11.894809</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>60.0</td>
      <td>70.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>CREATININE_PHOSPHOKINASE</th>
      <td>299.0</td>
      <td>581.839465</td>
      <td>970.287881</td>
      <td>23.0</td>
      <td>116.5</td>
      <td>250.0</td>
      <td>582.0</td>
      <td>7861.0</td>
    </tr>
    <tr>
      <th>EJECTION_FRACTION</th>
      <td>299.0</td>
      <td>38.083612</td>
      <td>11.834841</td>
      <td>14.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>PLATELETS</th>
      <td>299.0</td>
      <td>263358.029264</td>
      <td>97804.236869</td>
      <td>25100.0</td>
      <td>212500.0</td>
      <td>262000.0</td>
      <td>303500.0</td>
      <td>850000.0</td>
    </tr>
    <tr>
      <th>SERUM_CREATININE</th>
      <td>299.0</td>
      <td>1.393880</td>
      <td>1.034510</td>
      <td>0.5</td>
      <td>0.9</td>
      <td>1.1</td>
      <td>1.4</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>SERUM_SODIUM</th>
      <td>299.0</td>
      <td>136.625418</td>
      <td>4.412477</td>
      <td>113.0</td>
      <td>134.0</td>
      <td>137.0</td>
      <td>140.0</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>TIME</th>
      <td>299.0</td>
      <td>130.260870</td>
      <td>77.614208</td>
      <td>4.0</td>
      <td>73.0</td>
      <td>115.0</td>
      <td>203.0</td>
      <td>285.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration 
# of the object and categorical variables
##################################
print('Categorical Variable Summary:')
display(heart_failure.describe(include=['category','object']).transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ANAEMIA</th>
      <td>299</td>
      <td>2</td>
      <td>Absent</td>
      <td>170</td>
    </tr>
    <tr>
      <th>DIABETES</th>
      <td>299</td>
      <td>2</td>
      <td>Absent</td>
      <td>174</td>
    </tr>
    <tr>
      <th>HIGH_BLOOD_PRESSURE</th>
      <td>299</td>
      <td>2</td>
      <td>Absent</td>
      <td>194</td>
    </tr>
    <tr>
      <th>SEX</th>
      <td>299</td>
      <td>2</td>
      <td>Male</td>
      <td>194</td>
    </tr>
    <tr>
      <th>SMOKING</th>
      <td>299</td>
      <td>2</td>
      <td>Absent</td>
      <td>203</td>
    </tr>
    <tr>
      <th>DEATH_EVENT</th>
      <td>299</td>
      <td>2</td>
      <td>0</td>
      <td>203</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed. All entries are unique.
2. No missing data noted for any variable with Null.Count>0 and Fill.Rate<1.0.
3. Low variance observed for two numeric predictors with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span>: First.Second.Mode.Ratio = 11.75
    * <span style="color: #FF0000">PLATELETS</span>: First.Second.Mode.Ratio = 6.25
4. No high skewness observed for the numeric predictor with Skewness>3 or Skewness<(-3).
   * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span>: Skewness = +4.46
    * <span style="color: #FF0000">SERUM_CREATININE</span>: Skewness = +4.46
5. No low variance observed for the numeric and categorical predictors with Unique.Count.Ratio>10.



```python
##################################
# Counting the number of duplicated rows
##################################
heart_failure.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(heart_failure.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(heart_failure.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(heart_failure)] * len(heart_failure.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(heart_failure.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(heart_failure.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ANAEMIA</td>
      <td>object</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DIABETES</td>
      <td>object</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EJECTION_FRACTION</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HIGH_BLOOD_PRESSURE</td>
      <td>object</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PLATELETS</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SERUM_CREATININE</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SERUM_SODIUM</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SEX</td>
      <td>object</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SMOKING</td>
      <td>object</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TIME</td>
      <td>float64</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>DEATH_EVENT</td>
      <td>category</td>
      <td>299</td>
      <td>299</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
print('Number of Columns with Missing Data:', str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
```

    Number of Columns with Missing Data: 0
    


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = heart_failure.index.values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(heart_failure.columns)] * len(heart_failure))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(heart_failure.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Exploring the rows
# for missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>294</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>295</th>
      <td>295</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>296</th>
      <td>296</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>297</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>298</td>
      <td>13</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Fill.Rate < 1.00
##################################
print('Number of Rows with Missing Data:',str(len(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])))
```

    Number of Rows with Missing Data: 0
    


```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
heart_failure_numeric = heart_failure.select_dtypes(include=['number','int'])
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = heart_failure_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = heart_failure_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = heart_failure_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = heart_failure_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = heart_failure_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [heart_failure[x].value_counts(dropna=True).index.tolist()[0] for x in heart_failure_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [heart_failure[x].value_counts(dropna=True).index.tolist()[1] for x in heart_failure_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [heart_failure_numeric[x].isin([heart_failure[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in heart_failure_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [heart_failure_numeric[x].isin([heart_failure[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in heart_failure_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = heart_failure_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(heart_failure_numeric)] * len(heart_failure_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = heart_failure_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = heart_failure_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>40.0</td>
      <td>60.833893</td>
      <td>60.0</td>
      <td>95.0</td>
      <td>60.00</td>
      <td>50.0</td>
      <td>33</td>
      <td>27</td>
      <td>1.222222</td>
      <td>47</td>
      <td>299</td>
      <td>0.157191</td>
      <td>0.423062</td>
      <td>-0.184871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>23.0</td>
      <td>581.839465</td>
      <td>250.0</td>
      <td>7861.0</td>
      <td>582.00</td>
      <td>66.0</td>
      <td>47</td>
      <td>4</td>
      <td>11.750000</td>
      <td>208</td>
      <td>299</td>
      <td>0.695652</td>
      <td>4.463110</td>
      <td>25.149046</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EJECTION_FRACTION</td>
      <td>14.0</td>
      <td>38.083612</td>
      <td>38.0</td>
      <td>80.0</td>
      <td>35.00</td>
      <td>38.0</td>
      <td>49</td>
      <td>40</td>
      <td>1.225000</td>
      <td>17</td>
      <td>299</td>
      <td>0.056856</td>
      <td>0.555383</td>
      <td>0.041409</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLATELETS</td>
      <td>25100.0</td>
      <td>263358.029264</td>
      <td>262000.0</td>
      <td>850000.0</td>
      <td>263358.03</td>
      <td>221000.0</td>
      <td>25</td>
      <td>4</td>
      <td>6.250000</td>
      <td>176</td>
      <td>299</td>
      <td>0.588629</td>
      <td>1.462321</td>
      <td>6.209255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SERUM_CREATININE</td>
      <td>0.5</td>
      <td>1.393880</td>
      <td>1.1</td>
      <td>9.4</td>
      <td>1.00</td>
      <td>1.1</td>
      <td>50</td>
      <td>32</td>
      <td>1.562500</td>
      <td>40</td>
      <td>299</td>
      <td>0.133779</td>
      <td>4.455996</td>
      <td>25.828239</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SERUM_SODIUM</td>
      <td>113.0</td>
      <td>136.625418</td>
      <td>137.0</td>
      <td>148.0</td>
      <td>136.00</td>
      <td>137.0</td>
      <td>40</td>
      <td>38</td>
      <td>1.052632</td>
      <td>27</td>
      <td>299</td>
      <td>0.090301</td>
      <td>-1.048136</td>
      <td>4.119712</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TIME</td>
      <td>4.0</td>
      <td>130.260870</td>
      <td>115.0</td>
      <td>285.0</td>
      <td>250.00</td>
      <td>187.0</td>
      <td>7</td>
      <td>7</td>
      <td>1.000000</td>
      <td>148</td>
      <td>299</td>
      <td>0.494983</td>
      <td>0.127803</td>
      <td>-1.212048</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    2




```python
##################################
# Identifying the numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>23.0</td>
      <td>581.839465</td>
      <td>250.0</td>
      <td>7861.0</td>
      <td>582.00</td>
      <td>66.0</td>
      <td>47</td>
      <td>4</td>
      <td>11.75</td>
      <td>208</td>
      <td>299</td>
      <td>0.695652</td>
      <td>4.463110</td>
      <td>25.149046</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLATELETS</td>
      <td>25100.0</td>
      <td>263358.029264</td>
      <td>262000.0</td>
      <td>850000.0</td>
      <td>263358.03</td>
      <td>221000.0</td>
      <td>25</td>
      <td>4</td>
      <td>6.25</td>
      <td>176</td>
      <td>299</td>
      <td>0.588629</td>
      <td>1.462321</td>
      <td>6.209255</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    2




```python
##################################
# Identifying the numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>23.0</td>
      <td>581.839465</td>
      <td>250.0</td>
      <td>7861.0</td>
      <td>582.0</td>
      <td>66.0</td>
      <td>47</td>
      <td>4</td>
      <td>11.7500</td>
      <td>208</td>
      <td>299</td>
      <td>0.695652</td>
      <td>4.463110</td>
      <td>25.149046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SERUM_CREATININE</td>
      <td>0.5</td>
      <td>1.393880</td>
      <td>1.1</td>
      <td>9.4</td>
      <td>1.0</td>
      <td>1.1</td>
      <td>50</td>
      <td>32</td>
      <td>1.5625</td>
      <td>40</td>
      <td>299</td>
      <td>0.133779</td>
      <td>4.455996</td>
      <td>25.828239</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object or categorical column only
##################################
heart_failure_object = heart_failure.select_dtypes(include=['object','category'])
```


```python
##################################
# Gathering the variable names for the object or categorical column
##################################
categorical_variable_name_list = heart_failure_object.columns
```


```python
##################################
# Gathering the first mode values for the object or categorical column
##################################
categorical_first_mode_list = [heart_failure[x].value_counts().index.tolist()[0] for x in heart_failure_object]
```


```python
##################################
# Gathering the second mode values for each object or categorical column
##################################
categorical_second_mode_list = [heart_failure[x].value_counts().index.tolist()[1] for x in heart_failure_object]
```


```python
##################################
# Gathering the count of first mode values for each object or categorical column
##################################
categorical_first_mode_count_list = [heart_failure_object[x].isin([heart_failure[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in heart_failure_object]
```


```python
##################################
# Gathering the count of second mode values for each object or categorical column
##################################
categorical_second_mode_count_list = [heart_failure_object[x].isin([heart_failure[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in heart_failure_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object or categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object or categorical column
##################################
categorical_unique_count_list = heart_failure_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object or categorical column
##################################
categorical_row_count_list = list([len(heart_failure_object)] * len(heart_failure_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object or categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                 categorical_first_mode_list,
                                                 categorical_second_mode_list,
                                                 categorical_first_mode_count_list,
                                                 categorical_second_mode_count_list,
                                                 categorical_first_second_mode_ratio_list,
                                                 categorical_unique_count_list,
                                                 categorical_row_count_list,
                                                 categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ANAEMIA</td>
      <td>Absent</td>
      <td>Present</td>
      <td>170</td>
      <td>129</td>
      <td>1.317829</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DIABETES</td>
      <td>Absent</td>
      <td>Present</td>
      <td>174</td>
      <td>125</td>
      <td>1.392000</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIGH_BLOOD_PRESSURE</td>
      <td>Absent</td>
      <td>Present</td>
      <td>194</td>
      <td>105</td>
      <td>1.847619</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SEX</td>
      <td>Male</td>
      <td>Female</td>
      <td>194</td>
      <td>105</td>
      <td>1.847619</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SMOKING</td>
      <td>Absent</td>
      <td>Present</td>
      <td>203</td>
      <td>96</td>
      <td>2.114583</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DEATH_EVENT</td>
      <td>0</td>
      <td>1</td>
      <td>203</td>
      <td>96</td>
      <td>2.114583</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object or categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of object or categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. Data transformation and scaling is necessary to address excessive outliers and high skewness as observed on several numeric predictors:
    * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span>: Skewness = +4.463, Outlier.Count = 29, Outlier.Ratio = 0.096
    * <span style="color: #FF0000">SERUM_CREATININE</span>: Skewness = +4.456, Outlier.Count = 29, Outlier Ratio = 0.096
    * <span style="color: #FF0000">PLATELETS</span>: Skewness = +1.462, Outlier.Count = 21, Outlier.Ratio = 0.070
2. Most variables achieved symmetrical distributions with minimal outliers after evaluating a Yeo-Johnson transformation, except for:
    * <span style="color: #FF0000">PLATELETS</span>: Skewness = +1.155, Outlier.Count = 18, Outlier.Ratio = 0.060
3. Among pairwise combinations of variables in the training subset, sufficiently high correlation values were observed but with no excessive multicollinearity noted:
    * <span style="color: #FF0000">TIME</span> and <span style="color: #FF0000">DEATH_EVENT</span>: Point.Biserial.Correlation = -0.530
    * <span style="color: #FF0000">SMOKING</span> and <span style="color: #FF0000">SEX</span>: Phi.Coefficient = +0.450
    * <span style="color: #FF0000">SERUM_CREATININE</span> and <span style="color: #FF0000">DEATH_EVENT</span>: Point.Biserial.Correlation = +0.290
    * <span style="color: #FF0000">AGE</span> and <span style="color: #FF0000">DEATH_EVENT</span>: Point.Biserial.Correlation = +0.250



```python
#################################
# Creating a dataset copy 
# for correlation analysis
##################################
heart_failure_correlation = heart_failure_original.copy()
display(heart_failure_correlation)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>DIABETES</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>TIME</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130.0</td>
      <td>1</td>
      <td>0</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136.0</td>
      <td>1</td>
      <td>0</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>62.0</td>
      <td>0</td>
      <td>61.0</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>155000.00</td>
      <td>1.1</td>
      <td>143.0</td>
      <td>1</td>
      <td>1</td>
      <td>270.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>295</th>
      <td>55.0</td>
      <td>0</td>
      <td>1820.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>270000.00</td>
      <td>1.2</td>
      <td>139.0</td>
      <td>0</td>
      <td>0</td>
      <td>271.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>296</th>
      <td>45.0</td>
      <td>0</td>
      <td>2060.0</td>
      <td>1</td>
      <td>60.0</td>
      <td>0</td>
      <td>742000.00</td>
      <td>0.8</td>
      <td>138.0</td>
      <td>0</td>
      <td>0</td>
      <td>278.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>45.0</td>
      <td>0</td>
      <td>2413.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>140000.00</td>
      <td>1.4</td>
      <td>140.0</td>
      <td>1</td>
      <td>1</td>
      <td>280.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>50.0</td>
      <td>0</td>
      <td>196.0</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>395000.00</td>
      <td>1.6</td>
      <td>136.0</td>
      <td>1</td>
      <td>1</td>
      <td>285.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 13 columns</p>
</div>



```python
##################################
# Initializing the correlation matrix
##################################
heart_failure_correlation_matrix = pd.DataFrame(np.zeros((len(heart_failure_correlation.columns), len(heart_failure_correlation.columns))),
                                              columns=heart_failure_correlation.columns,
                                              index=heart_failure_correlation.columns)
```


```python
##################################
# Calculating different types
# of correlation coefficients
# per variable type
##################################
for i in range(len(heart_failure_correlation.columns)):
    for j in range(i, len(heart_failure_correlation.columns)):
        if i == j:
            heart_failure_correlation_matrix.iloc[i, j] = 1.0
        else:
            if heart_failure_correlation.dtypes.iloc[i] == 'float64' and heart_failure_correlation.dtypes.iloc[j] == 'float64':
                # Pearson correlation for two continuous variables
                corr = heart_failure_correlation.iloc[:, i].corr(heart_failure_correlation.iloc[:, j])
            elif heart_failure_correlation.dtypes.iloc[i] == 'int64' or heart_failure_correlation.dtypes.iloc[j] == 'int64':
                # Point-biserial correlation for one continuous and one binary variable
                continuous_var = heart_failure_correlation.iloc[:, i] if heart_failure_correlation.dtypes.iloc[i] == 'int64' else heart_failure_correlation.iloc[:, j]
                binary_var = heart_failure_correlation.iloc[:, j] if heart_failure_correlation.dtypes.iloc[j] == 'int64' else heart_failure_correlation.iloc[:, i]
                corr, _ = pointbiserialr(continuous_var, binary_var)
            else:
                # Phi coefficient for two binary variables
                corr = heart_failure_correlation.iloc[:, i].corr(heart_failure_correlation.iloc[:, j])
            heart_failure_correlation_matrix.iloc[i, j] = corr
            heart_failure_correlation_matrix.iloc[j, i] = corr
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric and categorical columns
##################################
plt.figure(figsize=(17, 8))
sns.heatmap(heart_failure_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
```


    
![png](output_79_0.png)
    



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
heart_failure_numeric = heart_failure.select_dtypes(include=['number','int'])
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = heart_failure_numeric.columns
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = heart_failure_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
heart_failure_numeric_q1 = heart_failure_numeric.quantile(0.25)
heart_failure_numeric_q3 = heart_failure_numeric.quantile(0.75)
heart_failure_numeric_iqr = heart_failure_numeric_q3 - heart_failure_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((heart_failure_numeric < (heart_failure_numeric_q1 - 1.5 * heart_failure_numeric_iqr)) | (heart_failure_numeric > (heart_failure_numeric_q3 + 1.5 * heart_failure_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(heart_failure_numeric)] * len(heart_failure_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>0.423062</td>
      <td>0</td>
      <td>299</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>4.463110</td>
      <td>29</td>
      <td>299</td>
      <td>0.096990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EJECTION_FRACTION</td>
      <td>0.555383</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLATELETS</td>
      <td>1.462321</td>
      <td>21</td>
      <td>299</td>
      <td>0.070234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SERUM_CREATININE</td>
      <td>4.455996</td>
      <td>29</td>
      <td>299</td>
      <td>0.096990</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SERUM_SODIUM</td>
      <td>-1.048136</td>
      <td>4</td>
      <td>299</td>
      <td>0.013378</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TIME</td>
      <td>0.127803</td>
      <td>0</td>
      <td>299</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in heart_failure_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=heart_failure_numeric, x=column)
```


    
![png](output_88_0.png)
    



    
![png](output_88_1.png)
    



    
![png](output_88_2.png)
    



    
![png](output_88_3.png)
    



    
![png](output_88_4.png)
    



    
![png](output_88_5.png)
    



    
![png](output_88_6.png)
    



```python
##################################
# Formulating the dataset
# with numeric predictor columns only
##################################
heart_failure_numeric_predictor = heart_failure_numeric.drop('TIME', axis=1)
```


```python
##################################
# Formulating the dataset
# with categorical or object columns only
##################################
heart_failure_categorical = heart_failure_original.select_dtypes(include=['category','object'])
```


```python
##################################
# Evaluating a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=True)
heart_failure_numeric_predictor_transformed_array = yeo_johnson_transformer.fit_transform(heart_failure_numeric_predictor)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
heart_failure_numeric_predictor_transformed = pd.DataFrame(heart_failure_numeric_predictor_transformed_array,
                                                           columns=heart_failure_numeric_predictor.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric predictor columns
##################################
for column in heart_failure_numeric_predictor_transformed:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=heart_failure_numeric_predictor_transformed, x=column)
```


    
![png](output_93_0.png)
    



    
![png](output_93_1.png)
    



    
![png](output_93_2.png)
    



    
![png](output_93_3.png)
    



    
![png](output_93_4.png)
    



    
![png](output_93_5.png)
    



```python
##################################
# Formulating the outlier summary
# for all numeric predictor columns
##################################
numeric_variable_name_list = heart_failure_numeric_predictor_transformed.columns
numeric_skewness_list = heart_failure_numeric_predictor_transformed.skew()
heart_failure_numeric_predictor_transformed_q1 = heart_failure_numeric_predictor_transformed.quantile(0.25)
heart_failure_numeric_predictor_transformed_q3 = heart_failure_numeric_predictor_transformed.quantile(0.75)
heart_failure_numeric_predictor_transformed_iqr = heart_failure_numeric_predictor_transformed_q3 - heart_failure_numeric_predictor_transformed_q1
numeric_outlier_count_list = ((heart_failure_numeric_predictor_transformed < (heart_failure_numeric_predictor_transformed_q1 - 1.5 * heart_failure_numeric_predictor_transformed_iqr)) | (heart_failure_numeric_predictor_transformed > (heart_failure_numeric_predictor_transformed_q3 + 1.5 * heart_failure_numeric_predictor_transformed_iqr))).sum()
numeric_row_count_list = list([len(heart_failure_numeric_predictor_transformed)] * len(heart_failure_numeric_predictor_transformed.columns))
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)

numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>-0.000746</td>
      <td>0</td>
      <td>299</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CREATININE_PHOSPHOKINASE</td>
      <td>0.044225</td>
      <td>0</td>
      <td>299</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EJECTION_FRACTION</td>
      <td>-0.006637</td>
      <td>2</td>
      <td>299</td>
      <td>0.006689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLATELETS</td>
      <td>0.155360</td>
      <td>18</td>
      <td>299</td>
      <td>0.060201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SERUM_CREATININE</td>
      <td>0.150380</td>
      <td>1</td>
      <td>299</td>
      <td>0.003344</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SERUM_SODIUM</td>
      <td>0.082305</td>
      <td>3</td>
      <td>299</td>
      <td>0.010033</td>
    </tr>
  </tbody>
</table>
</div>


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. In the estimated baseline survival plot, the survival probability did not reach 50% over the observed time period. The last observed survival probability was 58% at <span style="color: #FF0000">TIME=258</span>. Therefore, the median survival time could not be determined from the current data. This suggests that the majority of individuals in the cohort maintained a survival probability above 50% throughout the follow-up period. .
2. Bivariate analysis identified individual predictors with potential association to the event status based on visual inspection.
    * Higher values for the following numeric predictors are associated with <span style="color: #FF0000">DEATH_EVENT=True</span>: 
        * <span style="color: #FF0000">AGE</span>
        * <span style="color: #FF0000">SERUM_CREATININE</span>
    * Lower values for the following numeric predictors are associated with <span style="color: #FF0000">DEATH_EVENT=True</span>: 
        * <span style="color: #FF0000">EJECTION_FRACTION</span>
        * <span style="color: #FF0000">SERUM_SODIUM</span>    
    * Higher counts for the following object predictors are associated with better differentiation between <span style="color: #FF0000">DEATH_EVENT=True</span> and <span style="color: #FF0000">DEATH_EVENT=False</span>:  
        * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
2. Bivariate analysis identified individual predictors with potential association to the survival time based on visual inspection.
    * No numeric predictors were associated with <span style="color: #FF0000">TIME</span>: 
    * Levels for the following object predictors are associated with differences in <span style="color: #FF0000">TIME</span> between <span style="color: #FF0000">DEATH_EVENT=True</span> and <span style="color: #FF0000">DEATH_EVENT=False</span>:  
        * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>



```python
##################################
# Formulating a complete dataframe
##################################
heart_failure_EDA = pd.concat([heart_failure_numeric_predictor_transformed,
                               heart_failure.select_dtypes(include=['category','object']),
                               heart_failure_numeric['TIME']],
                              axis=1)
heart_failure_EDA['DEATH_EVENT'] = heart_failure_EDA['DEATH_EVENT'].replace({0: False, 1: True})
heart_failure_EDA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>EJECTION_FRACTION</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>DIABETES</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>DEATH_EVENT</th>
      <th>TIME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.173233</td>
      <td>0.691615</td>
      <td>-1.773346</td>
      <td>0.110528</td>
      <td>1.212227</td>
      <td>-1.468519</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Male</td>
      <td>Absent</td>
      <td>True</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.423454</td>
      <td>2.401701</td>
      <td>0.100914</td>
      <td>0.093441</td>
      <td>-0.087641</td>
      <td>-0.244181</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Male</td>
      <td>Absent</td>
      <td>True</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.434332</td>
      <td>-0.553424</td>
      <td>-1.773346</td>
      <td>-1.093142</td>
      <td>0.381817</td>
      <td>-1.642143</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Male</td>
      <td>Present</td>
      <td>True</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.910411</td>
      <td>-0.833885</td>
      <td>-1.773346</td>
      <td>-0.494713</td>
      <td>1.212227</td>
      <td>-0.006503</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Male</td>
      <td>Absent</td>
      <td>True</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.434332</td>
      <td>-0.462335</td>
      <td>-1.773346</td>
      <td>0.720277</td>
      <td>1.715066</td>
      <td>-3.285073</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Female</td>
      <td>Absent</td>
      <td>True</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the baseline survival curve
# and computing the survival rates
##################################
kmf = KaplanMeierFitter()
kmf.fit(durations=heart_failure_EDA['TIME'], event_observed=heart_failure_EDA['DEATH_EVENT'])
plt.figure(figsize=(17, 8))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Baseline Survival Plot')
plt.ylim(0, 1.05)
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')

##################################
# Determing the at-risk numbers
##################################
at_risk_counts = kmf.event_table.at_risk
survival_probabilities = kmf.survival_function_.values.flatten()
time_points = kmf.survival_function_.index
for time, prob, at_risk in zip(time_points, survival_probabilities, at_risk_counts):
    if time % 50 == 0: 
        plt.text(time, prob, f'{prob:.2f} : {at_risk}', ha='left', fontsize=10)
median_survival_time = kmf.median_survival_time_
plt.axvline(x=median_survival_time, color='r', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()
```


    
![png](output_98_0.png)
    



```python
##################################
# Computing the median survival time
##################################
median_survival_time = kmf.median_survival_time_
print(f'Median Survival Time: {median_survival_time}')
```

    Median Survival Time: inf
    


```python
##################################
# Exploring the relationships between
# the numeric predictors and event status
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    sns.boxplot(x='DEATH_EVENT', y=heart_failure_numeric_predictor.columns[i-1], hue='DEATH_EVENT', data=heart_failure_EDA)
    plt.title(f'{heart_failure_numeric_predictor.columns[i-1]} vs DEATH_EVENT Status')
    plt.legend(loc='upper center')
plt.tight_layout()
plt.show()
```


    
![png](output_100_0.png)
    



```python
##################################
# Exploring the relationships between
# the numeric predictors and event status
##################################
heart_failure_categorical_predictor = heart_failure_categorical.drop('DEATH_EVENT',axis=1)
heart_failure_EDA[int_columns] = heart_failure_EDA[int_columns].astype(object)
plt.figure(figsize=(17, 12))
for i in range(1, 6):
    plt.subplot(2, 3, i)
    sns.countplot(hue='DEATH_EVENT', x=heart_failure_categorical_predictor.columns[i-1], data=heart_failure_EDA)
    plt.title(f'{heart_failure_categorical_predictor.columns[i-1]} vs DEATH_EVENT Status')
    plt.legend(loc='upper center')
plt.tight_layout()
plt.show()
```


    
![png](output_101_0.png)
    



```python
##################################
# Exploring the relationships between
# the numeric predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    sns.scatterplot(x='TIME', y=heart_failure_numeric_predictor.columns[i-1], hue='DEATH_EVENT', data=heart_failure_EDA)
    loess_smoothed = lowess(heart_failure_EDA['TIME'], heart_failure_EDA[heart_failure_numeric_predictor.columns[i-1]], frac=0.3)
    plt.plot(loess_smoothed[:, 1], loess_smoothed[:, 0], color='red')
    plt.title(f'{heart_failure_numeric_predictor.columns[i-1]} vs Survival Time')
    plt.legend(loc='upper center')
plt.tight_layout()
plt.show()
```


    
![png](output_102_0.png)
    



```python
##################################
# Exploring the relationships between
# the object predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 6):
    plt.subplot(2, 3, i)
    sns.boxplot(x=heart_failure_categorical_predictor.columns[i-1], y='TIME', hue='DEATH_EVENT', data=heart_failure_EDA)
    plt.title(f'{heart_failure_categorical_predictor.columns[i-1]} vs Survival Time')
    plt.legend(loc='upper center')
plt.tight_layout()
plt.show()
```


    
![png](output_103_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">DEATH_EVENT</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups True and False is equal to zero  
    * **Alternative**: Difference in the means between groups True and False is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from the <span style="color: #FF0000">Status</span> groups in 4 numeric predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">SERUM_CREATININE</span>: T.Test.Statistic=-6.825, T.Test.PValue=0.000
    * <span style="color: #FF0000">EJECTION_FRACTION</span>: T.Test.Statistic=+5.495, T.Test.PValue=0.000 
    * <span style="color: #FF0000">AGE</span>: T.Test.Statistic=-4.274, T.Test.PValue=0.000  
    * <span style="color: #FF0000">SERUM_SODIUM</span>: T.Test.Statistic=+3.229, T.Test.PValue=0.001 
3. The relationship between the object predictors to the <span style="color: #FF0000">DEATH_EVENT</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: The object predictor is independent of the event variable 
    * **Alternative**: The object predictor is dependent on the event variable   
4. There were no categorical predictors that demonstrated sufficient evidence to conclude of a statistically significant relationship between the individual categories and the <span style="color: #FF0000">Status</span> groups with high chisquare statistic values with reported low p-values less than the significance level of 0.05.
5. The relationship between the object predictors to the <span style="color: #FF0000">DEATH_EVENT</span> and <span style="color: #FF0000">TIME</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the object predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the object predictor.
6. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">DEATH_EVENT</span> groups with respect to the survival duration <span style="color: #FF0000">TIME</span> in 1 categorical predictor given its high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>: LR.Test.Statistic=4.406, LR.Test.PValue=0.035
7. The relationship between the binned numeric predictors to the <span style="color: #FF0000">DEATH_EVENT</span> and <span style="color: #FF0000">TIME</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
8. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">DEATH_EVENT</span> groups with respect to the survival duration <span style="color: #FF0000">TIME</span> in 9 binned numeric predictors given their high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Binned_SERUM_CREATININE</span>: LR.Test.Statistic=21.190, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_EJECTION_FRACTION</span>: LR.Test.Statistic=9.469, LR.Test.PValue=0.002 
    * <span style="color: #FF0000">Binned_AGE</span>: LR.Test.Statistic=4.951, LR.Test.PValue=0.026
    * <span style="color: #FF0000">Binned_SERUM_SODIUM</span>: LR.Test.Statistic=4.887, LR.Test.PValue=0.027
      


```python
##################################
# Formulating a complete dataframe
##################################
heart_failure_HT = pd.concat([heart_failure_numeric_predictor_transformed,
                               heart_failure_categorical,
                               heart_failure_numeric['TIME']],
                              axis=1)
heart_failure_HT['DEATH_EVENT'] = heart_failure_HT['DEATH_EVENT'].replace({0: False, 1: True})
heart_failure_HT.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>EJECTION_FRACTION</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>DIABETES</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>DEATH_EVENT</th>
      <th>TIME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.173233</td>
      <td>0.691615</td>
      <td>-1.773346</td>
      <td>0.110528</td>
      <td>1.212227</td>
      <td>-1.468519</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>True</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.423454</td>
      <td>2.401701</td>
      <td>0.100914</td>
      <td>0.093441</td>
      <td>-0.087641</td>
      <td>-0.244181</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>True</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.434332</td>
      <td>-0.553424</td>
      <td>-1.773346</td>
      <td>-1.093142</td>
      <td>0.381817</td>
      <td>-1.642143</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.910411</td>
      <td>-0.833885</td>
      <td>-1.773346</td>
      <td>-0.494713</td>
      <td>1.212227</td>
      <td>-0.006503</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>True</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.434332</td>
      <td>-0.462335</td>
      <td>-1.773346</td>
      <td>0.720277</td>
      <td>1.715066</td>
      <td>-3.285073</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Computing the t-test 
# statistic and p-values
# between the event variable
# and numeric predictor columns
##################################
heart_failure_numeric_ttest_event = {}
for numeric_column in heart_failure_numeric_predictor.columns:
    group_0 = heart_failure_HT[heart_failure_HT.loc[:,'DEATH_EVENT']==False]
    group_1 = heart_failure_HT[heart_failure_HT.loc[:,'DEATH_EVENT']==True]
    heart_failure_numeric_ttest_event['DEATH_EVENT_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the event variable
# and numeric predictor columns
##################################
heart_failure_numeric_ttest_summary = heart_failure_HT.from_dict(heart_failure_numeric_ttest_event, orient='index')
heart_failure_numeric_ttest_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(heart_failure_numeric_ttest_summary.sort_values(by=['T.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DEATH_EVENT_SERUM_CREATININE</th>
      <td>-6.825678</td>
      <td>4.927143e-11</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_EJECTION_FRACTION</th>
      <td>5.495673</td>
      <td>8.382875e-08</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_AGE</th>
      <td>-4.274623</td>
      <td>2.582635e-05</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_SERUM_SODIUM</th>
      <td>3.229580</td>
      <td>1.378737e-03</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_PLATELETS</th>
      <td>1.031261</td>
      <td>3.032576e-01</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_CREATININE_PHOSPHOKINASE</th>
      <td>-0.565564</td>
      <td>5.721174e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the event variable
# and categorical predictor columns
##################################
heart_failure_categorical_chisquare_event = {}
for categorical_column in heart_failure_categorical_predictor.columns:
    contingency_table = pd.crosstab(heart_failure_HT[categorical_column], 
                                    heart_failure_HT['DEATH_EVENT'])
    heart_failure_categorical_chisquare_event['DEATH_EVENT_' + categorical_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the event variable
# and categorical predictor columns
##################################
heart_failure_categorical_chisquare_event_summary = heart_failure_HT.from_dict(heart_failure_categorical_chisquare_event, orient='index')
heart_failure_categorical_chisquare_event_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(heart_failure_categorical_chisquare_event_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DEATH_EVENT_HIGH_BLOOD_PRESSURE</th>
      <td>1.543461</td>
      <td>0.214103</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_ANAEMIA</th>
      <td>1.042175</td>
      <td>0.307316</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_SMOKING</th>
      <td>0.007331</td>
      <td>0.931765</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_DIABETES</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_SEX</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Exploring the relationships between
# the categorical predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 18))
for i in range(1, 6):
    ax = plt.subplot(3, 2, i)
    for group in [0,1]:
        kmf.fit(durations=heart_failure_HT[heart_failure_HT[heart_failure_categorical_predictor.columns[i-1]] == group]['TIME'],
                event_observed=heart_failure_HT[heart_failure_HT[heart_failure_categorical_predictor.columns[i-1]] == group]['DEATH_EVENT'], label=group)
        kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {heart_failure_categorical_predictor.columns[i-1]} Categories')
    plt.xlabel('TIME')
    plt.ylabel('DEATH_EVENT Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_110_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the categorical predictor columns
##################################
heart_failure_categorical_lrtest_event = {}
for categorical_column in heart_failure_categorical_predictor.columns:
    groups = [0,1]
    group_0_event = heart_failure_HT[heart_failure_HT[categorical_column] == groups[0]]['DEATH_EVENT']
    group_1_event = heart_failure_HT[heart_failure_HT[categorical_column] == groups[1]]['DEATH_EVENT']
    group_0_duration = heart_failure_HT[heart_failure_HT[categorical_column] == groups[0]]['TIME']
    group_1_duration = heart_failure_HT[heart_failure_HT[categorical_column] == groups[1]]['TIME']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    heart_failure_categorical_lrtest_event['DEATH_EVENT_TIME_' + categorical_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the categorical predictor columns
##################################
heart_failure_categorical_lrtest_summary = heart_failure_HT.from_dict(heart_failure_categorical_lrtest_event, orient='index')
heart_failure_categorical_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(heart_failure_categorical_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DEATH_EVENT_TIME_HIGH_BLOOD_PRESSURE</th>
      <td>4.406248</td>
      <td>0.035808</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_ANAEMIA</th>
      <td>2.726464</td>
      <td>0.098698</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_DIABETES</th>
      <td>0.040528</td>
      <td>0.840452</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_SEX</th>
      <td>0.003971</td>
      <td>0.949752</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_SMOKING</th>
      <td>0.002042</td>
      <td>0.963960</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating an alternate copy of the 
# EDA data which will utilize
# binning for numeric predictors
##################################
heart_failure_HT_binned = heart_failure_HT.copy()

##################################
# Creating a function to bin
# numeric predictors into two groups
##################################
def bin_numeric_predictor(df, predictor):
    median = df[predictor].median()
    df[f'Binned_{predictor}'] = np.where(df[predictor] <= median, 0, 1)
    return df

##################################
# Binning the numeric predictors
# in the alternate data into two groups
##################################
for numeric_column in heart_failure_numeric_predictor.columns:
    heart_failure_HT_binned = bin_numeric_predictor(heart_failure_HT_binned, numeric_column)
    
##################################
# Formulating the binned numeric predictors
##################################    
heart_failure_binned_numeric_predictor = ["Binned_" + predictor for predictor in heart_failure_numeric_predictor.columns]
```


```python
##################################
# Exploring the relationships between
# the binned numeric predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 18))
for i in range(1, 7):
    ax = plt.subplot(3, 2, i)
    for group in [0,1]:
            kmf.fit(durations=heart_failure_HT_binned[heart_failure_HT_binned[heart_failure_binned_numeric_predictor[i-1]] == group]['TIME'],
                event_observed=heart_failure_HT_binned[heart_failure_HT_binned[heart_failure_binned_numeric_predictor[i-1]] == group]['DEATH_EVENT'], label=group)
            kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {heart_failure_binned_numeric_predictor[i-1]} Categories')
    plt.xlabel('TIME')
    plt.ylabel('DEATH_EVENT Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_114_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the binned numeric predictor columns
##################################
heart_failure_binned_numeric_lrtest_event = {}
for binned_numeric_column in heart_failure_binned_numeric_predictor:
    groups = [0,1]
    group_0_event = heart_failure_HT_binned[heart_failure_HT_binned[binned_numeric_column] == groups[0]]['DEATH_EVENT']
    group_1_event = heart_failure_HT_binned[heart_failure_HT_binned[binned_numeric_column] == groups[1]]['DEATH_EVENT']
    group_0_duration = heart_failure_HT_binned[heart_failure_HT_binned[binned_numeric_column] == groups[0]]['TIME']
    group_1_duration = heart_failure_HT_binned[heart_failure_HT_binned[binned_numeric_column] == groups[1]]['TIME']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    heart_failure_binned_numeric_lrtest_event['DEATH_EVENT_TIME_' + binned_numeric_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the binned numeric predictor columns
##################################
heart_failure_binned_numeric_lrtest_summary = heart_failure_HT_binned.from_dict(heart_failure_binned_numeric_lrtest_event, orient='index')
heart_failure_binned_numeric_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(heart_failure_binned_numeric_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_SERUM_CREATININE</th>
      <td>21.190414</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_EJECTION_FRACTION</th>
      <td>9.469633</td>
      <td>0.002089</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_AGE</th>
      <td>4.951760</td>
      <td>0.026064</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_SERUM_SODIUM</th>
      <td>4.887878</td>
      <td>0.027046</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_CREATININE_PHOSPHOKINASE</th>
      <td>0.055576</td>
      <td>0.813630</td>
    </tr>
    <tr>
      <th>DEATH_EVENT_TIME_Binned_PLATELETS</th>
      <td>0.009122</td>
      <td>0.923912</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Predictive Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Pre-Modelling Data Preparation <a class="anchor" id="1.6.1"></a>

1. All dichotomous categorical predictors and the target variable were one-hot encoded for the downstream modelling process. 
2. Predictors determined with insufficient association with the <span style="color: #FF0000">DEATH_EVENT</span> and <span style="color: #FF0000">TIME</span> variables were excluded for the subsequent modelling steps.
    * <span style="color: #FF0000">DIABETES</span>: LR.Test.Statistic=0.040, LR.Test.PValue=0.840
    * <span style="color: #FF0000">SEX</span>: LR.Test.Statistic=0.003, LR.Test.PValue=0.949
    * <span style="color: #FF0000">SMOKING</span>: LR.Test.Statistic=0.002, LR.Test.PValue=0.963  
    * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span>: LR.Test.Statistic=0.055, LR.Test.PValue=0.813
    * <span style="color: #FF0000">PLATELETS</span>: LR.Test.Statistic=0.009, LR.Test.PValue=0.923



```python
#################################
# Creating a dataset copy 
# for data splitting and modelling
##################################
heart_failure_transformed = heart_failure_original.copy()
heart_failure_transformed['DEATH_EVENT'] = heart_failure_transformed['DEATH_EVENT'].replace({0: False, 1: True})
display(heart_failure_transformed)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>CREATININE_PHOSPHOKINASE</th>
      <th>DIABETES</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>PLATELETS</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>SEX</th>
      <th>SMOKING</th>
      <th>TIME</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130.0</td>
      <td>1</td>
      <td>0</td>
      <td>4.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136.0</td>
      <td>1</td>
      <td>0</td>
      <td>6.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>62.0</td>
      <td>0</td>
      <td>61.0</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>155000.00</td>
      <td>1.1</td>
      <td>143.0</td>
      <td>1</td>
      <td>1</td>
      <td>270.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>295</th>
      <td>55.0</td>
      <td>0</td>
      <td>1820.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>270000.00</td>
      <td>1.2</td>
      <td>139.0</td>
      <td>0</td>
      <td>0</td>
      <td>271.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>296</th>
      <td>45.0</td>
      <td>0</td>
      <td>2060.0</td>
      <td>1</td>
      <td>60.0</td>
      <td>0</td>
      <td>742000.00</td>
      <td>0.8</td>
      <td>138.0</td>
      <td>0</td>
      <td>0</td>
      <td>278.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>297</th>
      <td>45.0</td>
      <td>0</td>
      <td>2413.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>140000.00</td>
      <td>1.4</td>
      <td>140.0</td>
      <td>1</td>
      <td>1</td>
      <td>280.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>298</th>
      <td>50.0</td>
      <td>0</td>
      <td>196.0</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>395000.00</td>
      <td>1.6</td>
      <td>136.0</td>
      <td>1</td>
      <td>1</td>
      <td>285.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 13 columns</p>
</div>



```python
##################################
# Saving the tranformed data
# to the DATASETS_PREPROCESSED_PATH
##################################
heart_failure_transformed.to_csv(os.path.join("..", DATASETS_PREPROCESSED_PATH, "heart_failure_transformed.csv"), index=False)
```


```python
##################################
# Filtering out predictors that did not exhibit 
# sufficient discrimination of the target variable
# Saving the tranformed data
# to the DATASETS_PREPROCESSED_PATH
##################################
heart_failure_filtered = heart_failure_transformed.drop(['DIABETES','SEX', 'SMOKING', 'CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
heart_failure_filtered.to_csv(os.path.join("..", DATASETS_FINAL_PATH, "heart_failure_final.csv"), index=False)
display(heart_failure_filtered)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>TIME</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>1</td>
      <td>1.9</td>
      <td>130.0</td>
      <td>4.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>1.1</td>
      <td>136.0</td>
      <td>6.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>129.0</td>
      <td>7.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0</td>
      <td>1.9</td>
      <td>137.0</td>
      <td>7.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0</td>
      <td>2.7</td>
      <td>116.0</td>
      <td>8.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>62.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>143.0</td>
      <td>270.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>295</th>
      <td>55.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>1.2</td>
      <td>139.0</td>
      <td>271.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>296</th>
      <td>45.0</td>
      <td>0</td>
      <td>60.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>138.0</td>
      <td>278.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>297</th>
      <td>45.0</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>140.0</td>
      <td>280.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>298</th>
      <td>50.0</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
      <td>285.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 8 columns</p>
</div>


### 1.6.2 Data Splitting <a class="anchor" id="1.6.2"></a>

1. The preprocessed dataset was divided into three subsets using a fixed random seed:
    * **test data**: 25% of the original data with class stratification applied
    * **train data (initial)**: 75% of the original data with class stratification applied
        * **train data (final)**: 75% of the **train (initial)** data with class stratification applied
        * **validation data**: 25% of the **train (initial)** data with class stratification applied
2. Although a moderate class imbalance between <span style="color: #FF0000">DEATH_EVENT=True</span> and <span style="color: #FF0000">DEATH_EVENT=False</span> was observed, maintining the time-to-even distribution is crucial for survival analysis. Resampling would require synthetic imputation of event times, which could introduce additional noise and bias into the model. Given the nature of the data, preserving the integrity of the time variable is of higher importance than correcting for a moderate class imbalance.
3. Models were developed from the **train data (final)**. Using the same dataset, a subset of models with optimal hyperparameters were selected, based on cross-validation.
4. Among candidate models with optimal hyperparameters, the final model were selected based on performance during **cross-validation** and **independent validation**. 
5. Performance of the selected final model (and other candidate models for post-model selection comparison) were evaluated using the **test data**. 
6. The preprocessed data is comprised of:
    * **299 rows** (observations)
        * **96 DEATH_EVENT=True**: 32.11%
        * **203 DEATH_EVENT=False**: 67.89%
    * **8 columns** (variables)
        * **2/8 event | duration** (object | numeric)
             * <span style="color: #FF0000">DEATH_EVENT</span>
             * <span style="color: #FF0000">TIME</span>
        * **4/8 predictor** (numeric)
             * <span style="color: #FF0000">AGE</span>
             * <span style="color: #FF0000">EJECTION_FRACTION</span>
             * <span style="color: #FF0000">SERUM_CREATININE</span>
             * <span style="color: #FF0000">SERUM_SODIUM</span>
        * **2/8 predictor** (object)
             * <span style="color: #FF0000">ANAEMIA </span>
             * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
7. The **train data (final)** subset is comprised of:
    * **168 rows** (observations)
        * **114 LUNG_CANCER=Yes**: 67.85%
        * **54 LUNG_CANCER=No**: 32.14%
    * **8 columns** (variables)
8. The **validation data** subset is comprised of:
    * **56 rows** (observations)
        * **38 LUNG_CANCER=Yes**: 67.85%
        * **18 LUNG_CANCER=No**: 32.14%
    * **8 columns** (variables)
9. The **test data** subset is comprised of:
    * **75 rows** (observations)
        * **51 DEATH_EVENT=True**: 68.93%
        * **24 DEATH_EVENT=False**: 32.07%
    * **8 columns** (variables)



```python
##################################
# Creating a dataset copy
# of the filtered data
##################################
heart_failure_final = heart_failure_filtered.copy()
```


```python
##################################
# Performing a general exploration
# of the final dataset
##################################
print('Final Dataset Dimensions: ')
display(heart_failure_final.shape)
```

    Final Dataset Dimensions: 
    


    (299, 8)



```python
print('Target Variable Breakdown: ')
heart_failure_breakdown = heart_failure_final.groupby('DEATH_EVENT', observed=True).size().reset_index(name='Count')
heart_failure_breakdown['Percentage'] = (heart_failure_breakdown['Count'] / len(heart_failure_final)) * 100
display(heart_failure_breakdown)
```

    Target Variable Breakdown: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DEATH_EVENT</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>203</td>
      <td>67.892977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>96</td>
      <td>32.107023</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the train and test data
# from the final dataset
# by applying stratification and
# using a 70-30 ratio
##################################
heart_failure_train_initial, heart_failure_test = train_test_split(heart_failure_final, 
                                                                   test_size=0.25, 
                                                                   stratify=heart_failure_final['DEATH_EVENT'], 
                                                                   random_state=88888888)
```


```python
##################################
# Performing a general exploration
# of the initial training dataset
##################################
X_train_initial = heart_failure_train_initial.drop(['DEATH_EVENT', 'TIME'], axis=1)
y_train_initial = heart_failure_train_initial[['DEATH_EVENT', 'TIME']]
print('Initial Training Dataset Dimensions: ')
display(X_train_initial.shape)
display(y_train_initial.shape)
print('Initial Training Target Variable Breakdown: ')
display(y_train_initial['DEATH_EVENT'].value_counts())
print('Initial Training Target Variable Proportion: ')
display(y_train_initial['DEATH_EVENT'].value_counts(normalize = True))
```

    Initial Training Dataset Dimensions: 
    


    (224, 6)



    (224, 2)


    Initial Training Target Variable Breakdown: 
    


    DEATH_EVENT
    False    152
    True      72
    Name: count, dtype: int64


    Initial Training Target Variable Proportion: 
    


    DEATH_EVENT
    False    0.678571
    True     0.321429
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the test dataset
##################################
X_test = heart_failure_test.drop(['DEATH_EVENT', 'TIME'], axis=1)
y_test = heart_failure_test[['DEATH_EVENT', 'TIME']]
print('Test Dataset Dimensions: ')
display(X_test.shape)
display(y_test.shape)
print('Test Target Variable Breakdown: ')
display(y_test['DEATH_EVENT'].value_counts())
print('Test Target Variable Proportion: ')
display(y_test['DEATH_EVENT'].value_counts(normalize = True))
```

    Test Dataset Dimensions: 
    


    (75, 6)



    (75, 2)


    Test Target Variable Breakdown: 
    


    DEATH_EVENT
    False    51
    True     24
    Name: count, dtype: int64


    Test Target Variable Proportion: 
    


    DEATH_EVENT
    False    0.68
    True     0.32
    Name: proportion, dtype: float64



```python
##################################
# Formulating the train and validation data
# from the train dataset
# by applying stratification and
# using a 70-30 ratio
##################################
heart_failure_train, heart_failure_validation = train_test_split(heart_failure_train_initial, 
                                                                 test_size=0.25, 
                                                                 stratify=heart_failure_train_initial['DEATH_EVENT'], 
                                                                 random_state=88888888)
```


```python
##################################
# Performing a general exploration
# of the final training dataset
##################################
X_train = heart_failure_train.drop(columns=['DEATH_EVENT', 'TIME'], axis=1)
y_train = heart_failure_train[['DEATH_EVENT', 'TIME']]
print('Final Training Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Final Training Target Variable Breakdown: ')
display(y_train['DEATH_EVENT'].value_counts())
print('Final Training Target Variable Proportion: ')
display(y_train['DEATH_EVENT'].value_counts(normalize = True))
```

    Final Training Dataset Dimensions: 
    


    (168, 6)



    (168, 2)


    Final Training Target Variable Breakdown: 
    


    DEATH_EVENT
    False    114
    True      54
    Name: count, dtype: int64


    Final Training Target Variable Proportion: 
    


    DEATH_EVENT
    False    0.678571
    True     0.321429
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the validation dataset
##################################
X_validation = heart_failure_validation.drop(columns=['DEATH_EVENT', 'TIME'], axis = 1)
y_validation = heart_failure_validation[['DEATH_EVENT', 'TIME']]
print('Validation Dataset Dimensions: ')
display(X_validation.shape)
display(y_validation.shape)
print('Validation Target Variable Breakdown: ')
display(y_validation['DEATH_EVENT'].value_counts())
print('Validation Target Variable Proportion: ')
display(y_validation['DEATH_EVENT'].value_counts(normalize = True))
```

    Validation Dataset Dimensions: 
    


    (56, 6)



    (56, 2)


    Validation Target Variable Breakdown: 
    


    DEATH_EVENT
    False    38
    True     18
    Name: count, dtype: int64


    Validation Target Variable Proportion: 
    


    DEATH_EVENT
    False    0.678571
    True     0.321429
    Name: proportion, dtype: float64



```python
##################################
# Saving the training data
# to the DATASETS_FINAL_TRAIN_PATH
# and DATASETS_FINAL_TRAIN_FEATURES_PATH
# and DATASETS_FINAL_TRAIN_TARGET_PATH
##################################
heart_failure_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_PATH, "heart_failure_train.csv"), index=False)
X_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"), index=False)
```


```python
##################################
# Saving the validation data
# to the DATASETS_FINAL_VALIDATION_PATH
# and DATASETS_FINAL_VALIDATION_FEATURE_PATH
# and DATASETS_FINAL_VALIDATION_TARGET_PATH
##################################
heart_failure_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_PATH, "heart_failure_validation.csv"), index=False)
X_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_FEATURES_PATH, "X_validation.csv"), index=False)
y_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_TARGET_PATH, "y_validation.csv"), index=False)
```


```python
##################################
# Saving the test data
# to the DATASETS_FINAL_TEST_PATH
# and DATASETS_FINAL_TEST_FEATURES_PATH
# and DATASETS_FINAL_TEST_TARGET_PATH
##################################
heart_failure_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_PATH, "heart_failure_test.csv"), index=False)
X_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_FEATURES_PATH, "X_test.csv"), index=False)
y_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_TARGET_PATH, "y_test.csv"), index=False)
```


```python
##################################
# Converting the event and duration variables
# for the train, validation and test sets
# to array as preparation for modeling
##################################
y_train_array = np.array([(row.DEATH_EVENT, row.TIME) for index, row in y_train.iterrows()], dtype=[('DEATH_EVENT', 'bool'), ('TIME', 'int')])
y_validation_array = np.array([(row.DEATH_EVENT, row.TIME) for index, row in y_validation.iterrows()], dtype=[('DEATH_EVENT', 'bool'), ('TIME', 'int')])
y_test_array = np.array([(row.DEATH_EVENT, row.TIME) for index, row in y_test.iterrows()], dtype=[('DEATH_EVENT', 'bool'), ('TIME', 'int')])
```

### 1.6.3 Modelling Pipeline Development <a class="anchor" id="1.6.3"></a>

### 1.6.3.1 Cox Proportional Hazards Regression <a class="anchor" id="1.6.3.1"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A modelling pipeline was implemented with the following steps:
    * [Yeo-johnson transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from the <mark style="background-color: #CCECFF"><b>sklearn.processing</b></mark> Python library API applied to the numeric predictors only. Categorical predictors were excluded from the transformation.
    * [Cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.linear_model</b></mark> Python library API with 1 hyperparameter:
        * <span style="color: #FF0000">alpha</span> = regularization parameter for ridge regression penalty made to vary between 0.00, 0.01, 0.10, 1.00, 10.0 and 100.00
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method for 5 repeats with optimal model performance determined using the concordance index. 



```python
##################################
# Defining the modelling pipeline
# using the Cox Proportional Hazards Regression Model
##################################
coxph_pipeline_preprocessor = ColumnTransformer(
    transformers=[
        # Applying PowerTransformer to numeric columns only
        ('numeric_predictors', PowerTransformer(method='yeo-johnson', standardize=True), ['AGE', 'EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM'])  
        # Keeping the categorical columns unchanged
    ], remainder='passthrough'
)
coxph_pipeline = Pipeline([
    ('yeo_johnson', coxph_pipeline_preprocessor),
    ('coxph', CoxPHSurvivalAnalysis())])
```


```python
##################################
# Defining the hyperparameters for grid search
##################################
coxph_hyperparameter_grid = {'coxph__alpha': [0.01, 0.10, 1.00, 10.00]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using concordance index as the model evaluation metric
##################################
coxph_grid_search = GridSearchCV(estimator=coxph_pipeline,
                                 param_grid=coxph_hyperparameter_grid,
                                 cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=88888888),
                                 return_train_score=False,
                                 n_jobs=-1,
                                 verbose=1)
```

### 1.6.3.2 Cox Net Survival <a class="anchor" id="1.6.3.2"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Net Survival](https://doi.org/10.18637/jss.v039.i05) is a regularized version of the Cox Proportional Hazards model, which incorporates both L1 (Lasso) and L2 (Ridge) penalties. The model is useful when dealing with high-dimensional data where the number of predictors can be larger than the number of observations. The elastic net penalty helps in both variable selection (via L1) and multicollinearity handling (via L2). As a method, it can handle high-dimensional data and perform variable selection. Additionally, it balances between L1 and L2 penalties, offering flexibility in modeling. However, the process requires tuning of penalty parameters, which can be computationally intensive. Additionally, interpretation is more complex due to the regularization terms. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the penalized partial likelihood function, incorporating both L1 (Lasso) and L2 (Ridge) penalties; application of regularization techniques to estimate the regression coefficients by maximizing the penalized log-partial likelihood; performing cross-validation to select optimal values for the penalty parameters (alpha and l1_ratio); and the calculation of the hazard function and survival function for new data using the estimated regularized coefficients.

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A modelling pipeline was implemented with the following steps:
    * [Yeo-johnson transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from the <mark style="background-color: #CCECFF"><b>sklearn.processing</b></mark> Python library API applied to the numeric predictors only. Categorical predictors were excluded from the transformation.
    * [Cox net survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.linear_model</b></mark> Python library API with 2 hyperparameters:
        * <span style="color: #FF0000">l1_ratio</span> = ElasticNet mixing parameter made to vary between 0.10, 0.50 and 1.00
        * <span style="color: #FF0000">alpha_min_ratio</span> = minimum alpha of the regularization path made to vary between 0.0001 and 0.01
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method for 5 repeats with optimal model performance determined using the concordance index. 


```python
##################################
# Defining the modelling pipeline
# using the cox net survival analysis model
##################################
coxns_pipeline_preprocessor = ColumnTransformer(
    transformers=[
        # Applying PowerTransformer to numeric columns only
        ('numeric_predictors', PowerTransformer(method='yeo-johnson', standardize=True), ['AGE', 'EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM'])  
        # Keeping the categorical columns unchanged
    ], remainder='passthrough'  
)
coxns_pipeline = Pipeline([
    ('yeo_johnson', coxns_pipeline_preprocessor),
    ('coxns', CoxnetSurvivalAnalysis())])
```


```python
##################################
# Defining the hyperparameters for grid search
##################################
coxns_hyperparameter_grid = {'coxns__l1_ratio': [0.10, 0.50, 1.00],
                             'coxns__alpha_min_ratio': [0.0001, 0.01],
                             'coxns__fit_baseline_model': [True]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using concordance index as the model evaluation metric
##################################
coxns_grid_search = GridSearchCV(estimator=coxns_pipeline,
                                 param_grid=coxns_hyperparameter_grid,
                                 cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=88888888),
                                 return_train_score=False,
                                 n_jobs=-1,
                                 verbose=1)
```

### 1.6.3.3 Survival Tree <a class="anchor" id="1.6.3.3"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Survival Trees](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476296) are non-parametric models that partition the data into subgroups (nodes) based on the values of predictor variables, creating a tree-like structure. The tree is built by recursively splitting the data at nodes where the differences in survival times between subgroups are maximized. Each terminal node represents a different survival function. The method have no assumptions about the underlying distribution of survival times, can capture interactions between variables naturally and applies an interpretable visual representation. However, the process can be prone to overfitting, especially with small datasets, and may be less accurate compared to ensemble methods like Random Survival Forest. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves recursively splitting the data at nodes to maximize the differences in survival times between subgroups with the splitting criteria often involving statistical tests (e.g., log-rank test); choosing the best predictor variable and split point at each node that maximizes the separation of survival times; continuously splitting until stopping criteria are met (e.g., minimum number of observations in a node, maximum tree depth); and estimating the survival function based on the survival times of the observations at each terminal node.

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A modelling pipeline was implemented with the following steps:
    * [Yeo-johnson transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from the <mark style="background-color: #CCECFF"><b>sklearn.processing</b></mark> Python library API applied to the numeric predictors only. Categorical predictors were excluded from the transformation.
    * [Survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html) from the <mark style="background-color: #CCECFF"><b>sksurv.tree</b></mark> Python library API with 2 hyperparameters:
        * <span style="color: #FF0000">min_samples_split</span> = minimum number of samples required to split an internal node made to vary between 10, 15 and 20
        * <span style="color: #FF0000">min_samples_leaf</span> = minimum number of samples required to be at a leaf node made to vary between 3 and 6
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method for 5 repeats with optimal model performance determined using the concordance index.



```python
##################################
# Defining the modelling pipeline
# using the survival tree model
##################################
stree_pipeline_preprocessor = ColumnTransformer(
    transformers=[
        # Applying PowerTransformer to numeric columns only
        ('numeric_predictors', PowerTransformer(method='yeo-johnson', standardize=True), ['AGE', 'EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM'])  
        # Keeping the categorical columns unchanged
    ], remainder='passthrough'  
)
stree_pipeline = Pipeline([
    ('yeo_johnson', stree_pipeline_preprocessor),
    ('stree', SurvivalTree())])
```


```python
##################################
# Defining the hyperparameters for grid search
##################################
stree_hyperparameter_grid = {'stree__min_samples_split': [10, 15, 20],
                             'stree__min_samples_leaf': [3, 6],
                             'stree__random_state': [88888888]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using concordance index as the model evaluation metric
##################################
stree_grid_search = GridSearchCV(estimator=stree_pipeline,
                                 param_grid=stree_hyperparameter_grid,
                                 cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=88888888),
                                 return_train_score=False,
                                 n_jobs=-1,
                                 verbose=1)
```

### 1.6.3.4 Random Survival Forest <a class="anchor" id="1.6.3.4"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Random Survival Forest](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Random-survival-forests/10.1214/08-AOAS169.full) is an ensemble method that builds multiple survival trees and averages their predictions. The model combines the predictions of multiple survival trees, each built on a bootstrap sample of the data and a random subset of predictors. It uses the concept of ensemble learning to improve predictive accuracy and robustness. As a method, it handles high-dimensional data and complex interactions between variables well; can be more accurate and robust than a single survival tree; and provides measures of variable importance. However, the process can be bomputationally intensive due to the need to build multiple trees, and may be less interpretable than single trees or parametric models like the Cox model. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves generating multiple bootstrap samples from the original dataset; building a survival tree by recursively splitting the data at nodes using a random subset of predictor variables for each bootstrap sample; combining the predictions of all survival trees to form the random survival forest and averaging the survival functions predicted by all trees in the forest to obtain the final survival function for new data.

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A modelling pipeline was implemented with the following steps:
    * [Yeo-johnson transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from the <mark style="background-color: #CCECFF"><b>sklearn.processing</b></mark> Python library API applied to the numeric predictors only. Categorical predictors were excluded from the transformation.
    * [Random survival forest model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html) from the <mark style="background-color: #CCECFF"><b>sksurv.ensemble</b></mark> Python library API with 2 hyperparameters:
        * <span style="color: #FF0000">n_estimators</span> = number of trees in the forest made to vary between 100, 200 and 300
        * <span style="color: #FF0000">min_samples_split</span> = minimum number of samples required to split an internal node made to vary between 10, 15 and 20
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method for 5 repeats with optimal model performance determined using the concordance index.



```python
##################################
# Defining the modelling pipeline
# using the random survival forest model
##################################
rsf_pipeline_preprocessor = ColumnTransformer(
    transformers=[
        # Applying PowerTransformer to numeric columns only
        ('numeric_predictors', PowerTransformer(method='yeo-johnson', standardize=True), ['AGE', 'EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM'])  
        # Keeping the categorical columns unchanged
    ], remainder='passthrough'  
)
rsf_pipeline = Pipeline([
    ('yeo_johnson', rsf_pipeline_preprocessor),
    ('rsf', RandomSurvivalForest())])
```


```python
##################################
# Defining the hyperparameters for grid search
##################################
rsf_hyperparameter_grid = {'rsf__n_estimators': [100, 200, 300],
                           'rsf__min_samples_split': [10, 15, 20],
                           'rsf__random_state': [88888888]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using concordance index as the model evaluation metric
##################################
rsf_grid_search = GridSearchCV(estimator=rsf_pipeline,
                               param_grid=rsf_hyperparameter_grid,
                               cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=88888888),
                               return_train_score=False,
                               n_jobs=-1,
                               verbose=1)
```

### 1.6.3.5 Gradient Boosted Survival <a class="anchor" id="1.6.3.5"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Gradient Boosted Survival](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) is an ensemble technique that builds a series of survival trees, where each tree tries to correct the errors of the previous one. The model uses boosting, a sequential technique where each new tree is fit to the residuals of the combined previous trees, and combines the predictions of all the trees to produce a final prediction. As a method, it has high predictive accuracy, the ability to model complex relationships, and reduces bias and variance compared to single-tree models. However, the process can even be more computationally intensive than Random Survival Forest, requires careful tuning of multiple hyperparameters, and makes interpretation challenging due to the complex nature of the model. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves starting with an initial prediction (often the median survival time or a simple model); calculating the residuals (errors) of the current model's predictions; fitting a survival tree to the residuals to learn the errors made by the current model; updating the current model by adding the new tree weighted by a learning rate parameter; repeating previous steps for a fixed number of iterations or until convergence; and summing the predictions of all trees in the sequence to obtain the final survival function for new data.

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A modelling pipeline was implemented with the following steps:
    * [Yeo-johnson transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from the <mark style="background-color: #CCECFF"><b>sklearn.processing</b></mark> Python library API applied to the numeric predictors only. Categorical predictors were excluded from the transformation.
    * [Gradient boosted survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.ensemble</b></mark> Python library API with 2 hyperparameters:
        * <span style="color: #FF0000">n_estimators</span> = number of regression trees to create made to vary between 100, 200 and 300
        * <span style="color: #FF0000">learning_rate</span> = shrinkage parameter for the contribution of each tree made to vary between 0.05, 0.10 and 0.15
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method for 5 repeats with optimal model performance determined using the concordance index.



```python
##################################
# Defining the modelling pipeline
# using the gradient boosted survival model
##################################
gbs_pipeline_preprocessor = ColumnTransformer(
    transformers=[
        # Applying PowerTransformer to numeric columns only
        ('numeric_predictors', PowerTransformer(method='yeo-johnson', standardize=True), ['AGE', 'EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM'])  
        # Keeping the categorical columns unchanged
    ], remainder='passthrough'  
)
gbs_pipeline = Pipeline([
    ('yeo_johnson', gbs_pipeline_preprocessor),
    ('gbs', GradientBoostingSurvivalAnalysis())])
```


```python
##################################
# Defining the hyperparameters for grid search
##################################
gbs_hyperparameter_grid = {'gbs__n_estimators': [100, 200, 300],
                           'gbs__learning_rate': [0.05, 0.10, 0.15],
                           'gbs__random_state': [88888888]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using concordance index as the model evaluation metric
##################################
gbs_grid_search = GridSearchCV(estimator=gbs_pipeline,
                               param_grid=gbs_hyperparameter_grid,
                               cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=88888888),
                               return_train_score=False,
                               n_jobs=-1,
                               verbose=1)
```

### 1.6.4 Cox Proportional Hazards Regression Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.4"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

1. The [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.linear_model</b></mark> Python library API was implemented. 
2. The model implementation used 1 hyperparameter:
    * <span style="color: #FF0000">alpha</span> = regularization parameter for ridge regression penalty made to vary between 0.00, 0.01, 0.10, 1.00, 10.0 and 100.00
3. Hyperparameter tuning was conducted using the 5-fold cross-validation method repeated 5 times with optimal model performance using the concordance index determined for: 
    * <span style="color: #FF0000">alpha</span> = 10.00
4. The cross-validated model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7073
5. The apparent model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7419
6. The independent validation model performance of the final model is summarized as follows:
    * **Concordance Index** = 0.7394
7. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of moderate model overfitting.
8. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
9. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.
    


```python
##################################
# Performing hyperparameter tuning
# through K-fold cross-validation
# using the Cox Proportional Hazards Regression Model
##################################
coxph_grid_search.fit(X_train, y_train_array)
```

    Fitting 25 folds for each of 4 candidates, totalling 100 fits
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;coxph&#x27;, CoxPHSurvivalAnalysis())]),
             n_jobs=-1, param_grid={&#x27;coxph__alpha&#x27;: [0.01, 0.1, 1.0, 10.0]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;coxph&#x27;, CoxPHSurvivalAnalysis())]),
             n_jobs=-1, param_grid={&#x27;coxph__alpha&#x27;: [0.01, 0.1, 1.0, 10.0]},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxph&#x27;, CoxPHSurvivalAnalysis(alpha=10.0))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CoxPHSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>CoxPHSurvivalAnalysis(alpha=10.0)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Summarizing the hyperparameter tuning 
# results from K-fold cross-validation
##################################
coxph_grid_search_results = pd.DataFrame(coxph_grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
coxph_grid_search_results.loc[:, ~coxph_grid_search_results.columns.str.endswith('_time')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_coxph__alpha</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>split6_test_score</th>
      <th>split7_test_score</th>
      <th>...</th>
      <th>split18_test_score</th>
      <th>split19_test_score</th>
      <th>split20_test_score</th>
      <th>split21_test_score</th>
      <th>split22_test_score</th>
      <th>split23_test_score</th>
      <th>split24_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>10.00</td>
      <td>{'coxph__alpha': 10.0}</td>
      <td>0.758410</td>
      <td>0.681115</td>
      <td>0.780612</td>
      <td>0.840000</td>
      <td>0.521429</td>
      <td>0.588957</td>
      <td>0.637821</td>
      <td>0.862500</td>
      <td>...</td>
      <td>0.62</td>
      <td>0.683019</td>
      <td>0.732143</td>
      <td>0.712264</td>
      <td>0.700272</td>
      <td>0.525641</td>
      <td>0.791667</td>
      <td>0.707318</td>
      <td>0.084671</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>{'coxph__alpha': 1.0}</td>
      <td>0.764526</td>
      <td>0.715170</td>
      <td>0.785714</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.616564</td>
      <td>0.628205</td>
      <td>0.850000</td>
      <td>...</td>
      <td>0.63</td>
      <td>0.675472</td>
      <td>0.736607</td>
      <td>0.669811</td>
      <td>0.694823</td>
      <td>0.521368</td>
      <td>0.802083</td>
      <td>0.701906</td>
      <td>0.086235</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.10</td>
      <td>{'coxph__alpha': 0.1}</td>
      <td>0.758410</td>
      <td>0.702786</td>
      <td>0.790816</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.625767</td>
      <td>0.631410</td>
      <td>0.854167</td>
      <td>...</td>
      <td>0.62</td>
      <td>0.679245</td>
      <td>0.709821</td>
      <td>0.669811</td>
      <td>0.697548</td>
      <td>0.529915</td>
      <td>0.796875</td>
      <td>0.701768</td>
      <td>0.085985</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>{'coxph__alpha': 0.01}</td>
      <td>0.758410</td>
      <td>0.702786</td>
      <td>0.790816</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.625767</td>
      <td>0.631410</td>
      <td>0.854167</td>
      <td>...</td>
      <td>0.62</td>
      <td>0.679245</td>
      <td>0.709821</td>
      <td>0.669811</td>
      <td>0.694823</td>
      <td>0.529915</td>
      <td>0.796875</td>
      <td>0.701134</td>
      <td>0.086022</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 30 columns</p>
</div>




```python
##################################
# Identifying the best model
##################################
coxph_best_model_train_cv = coxph_grid_search.best_estimator_
print('Best Cox Proportional Hazards Regression Model using the Cross-Validated Train Data: ')
print(f"Best Model Parameters: {coxph_grid_search.best_params_}")
```

    Best Cox Proportional Hazards Regression Model using the Cross-Validated Train Data: 
    Best Model Parameters: {'coxph__alpha': 10.0}
    


```python
##################################
# Obtaining the cross-validation model performance of the 
# optimal Cox Proportional Hazards Regression Model
# on the train set
##################################
optimal_coxph_heart_failure_y_crossvalidation_ci = coxph_grid_search.best_score_
print(f"Cross-Validation Concordance Index: {optimal_coxph_heart_failure_y_crossvalidation_ci}")
```

    Cross-Validation Concordance Index: 0.7073178688853099
    


```python
##################################
# Formulating a Cox Proportional Hazards Regression Model
# with optimal hyperparameters
##################################
optimal_coxph_model = coxph_grid_search.best_estimator_
optimal_coxph_model.fit(X_train, y_train_array)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxph&#x27;, CoxPHSurvivalAnalysis(alpha=10.0))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxph&#x27;, CoxPHSurvivalAnalysis(alpha=10.0))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CoxPHSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>CoxPHSurvivalAnalysis(alpha=10.0)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Measuring model performance of the 
# optimal Cox Proportional Hazards Regression Model
# on the train set
##################################
optimal_coxph_heart_failure_y_train_pred = optimal_coxph_model.predict(X_train)
optimal_coxph_heart_failure_y_train_ci = concordance_index_censored(y_train_array['DEATH_EVENT'], 
                                                                    y_train_array['TIME'], 
                                                                    optimal_coxph_heart_failure_y_train_pred)[0]
print(f"Apparent Concordance Index: {optimal_coxph_heart_failure_y_train_ci}")
```

    Apparent Concordance Index: 0.7419406319821258
    


```python
##################################
# Measuring model performance of the 
# optimal Cox Proportional Hazards Regression Model
# on the validation set
##################################
optimal_coxph_heart_failure_y_validation_pred = optimal_coxph_model.predict(X_validation)
optimal_coxph_heart_failure_y_validation_ci = concordance_index_censored(y_validation_array['DEATH_EVENT'], 
                                                                         y_validation_array['TIME'], 
                                                                         optimal_coxph_heart_failure_y_validation_pred)[0]
print(f"Validation Concordance Index: {optimal_coxph_heart_failure_y_validation_ci}")
```

    Validation Concordance Index: 0.7394270122783083
    


```python
##################################
# Gathering the concordance indices
# from the train and tests sets for 
# Cox Proportional Hazards Regression Model
##################################
coxph_set = pd.DataFrame(["Train","Cross-Validation","Validation"])
coxph_ci_values = pd.DataFrame([optimal_coxph_heart_failure_y_train_ci,
                                optimal_coxph_heart_failure_y_crossvalidation_ci,
                                optimal_coxph_heart_failure_y_validation_ci])
coxph_method = pd.DataFrame(["COXPH"]*3)
coxph_summary = pd.concat([coxph_set, 
                           coxph_ci_values,
                           coxph_method], axis=1)
coxph_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_summary.reset_index(inplace=True, drop=True)
display(coxph_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.741941</td>
      <td>COXPH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.707318</td>
      <td>COXPH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>0.739427</td>
      <td>COXPH</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
heart_failure_validation.reset_index(drop=True, inplace=True)
kmf = KaplanMeierFitter()
heart_failure_validation['Predicted_Risks_CoxPH'] = optimal_coxph_heart_failure_y_validation_pred
heart_failure_validation['Predicted_RiskGroups_CoxPH'] = risk_groups = pd.qcut(heart_failure_validation['Predicted_Risks_CoxPH'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = heart_failure_validation[risk_groups == group]
    kmf.fit(group_data['TIME'], event_observed=group_data['DEATH_EVENT'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH Survival Probabilities by Predicted Risk Groups on Validation Set')
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')
plt.show()
```


    
![png](output_166_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
validation_case_details = X_validation.iloc[[5, 10, 15, 20, 25]]
display(validation_case_details)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>60.0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>42.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50.0</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>57.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45.0</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>127.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Gathering the event and duration information
# for 5 test case samples
##################################
print(y_validation_array[[5, 10, 15, 20, 25]])
```

    [(False, 258) ( True,  65) (False,  90) (False,  79) ( True,  14)]
    


```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(heart_failure_validation.loc[[5, 10, 15, 20, 25]][['Predicted_RiskGroups_CoxPH']])
```

       Predicted_RiskGroups_CoxPH
    5                    Low-Risk
    10                  High-Risk
    15                  High-Risk
    20                  High-Risk
    25                  High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 validation cases
##################################
validation_case = X_validation.iloc[[5, 10, 15, 20, 25]]
validation_case_labels = ['Patient_5','Patient_10','Patient_15','Patient_20','Patient_25',]
validation_case_cumulative_hazard_function = optimal_coxph_model.predict_cumulative_hazard_function(validation_case)
validation_case_survival_function = optimal_coxph_model.predict_survival_function(validation_case)

fig, ax = plt.subplots(1,2,figsize=(17, 8))
for hazard_prediction, survival_prediction in zip(validation_case_cumulative_hazard_function, validation_case_survival_function):
    ax[0].step(hazard_prediction.x,hazard_prediction(hazard_prediction.x),where='post')
    ax[1].step(survival_prediction.x,survival_prediction(survival_prediction.x),where='post')
ax[0].set_title('COXPH Cumulative Hazard for 5 Validation Cases')
ax[0].set_xlabel('TIME')
ax[0].set_ylim(0,2)
ax[0].set_ylabel('Cumulative Hazard')
ax[0].legend(validation_case_labels, loc="upper left")
ax[1].set_title('COXPH Survival Function for 5 Validation Cases')
ax[1].set_xlabel('TIME')
ax[1].set_ylabel('DEATH_EVENT Survival Probability')
ax[1].legend(validation_case_labels, loc="lower left")
plt.show()
```


    
![png](output_170_0.png)
    



```python
##################################
# Saving the best Cox Proportional Hazards Regression Model
# developed from the original training data
################################## 
joblib.dump(coxph_best_model_train_cv, 
            os.path.join("..", MODELS_PATH, "coxph_best_model.pkl"))
```




    ['..\\models\\coxph_best_model.pkl']



### 1.6.5 Cox Net Survival Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.5"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Net Survival](https://doi.org/10.18637/jss.v039.i05) is a regularized version of the Cox Proportional Hazards model, which incorporates both L1 (Lasso) and L2 (Ridge) penalties. The model is useful when dealing with high-dimensional data where the number of predictors can be larger than the number of observations. The elastic net penalty helps in both variable selection (via L1) and multicollinearity handling (via L2). As a method, it can handle high-dimensional data and perform variable selection. Additionally, it balances between L1 and L2 penalties, offering flexibility in modeling. However, the process requires tuning of penalty parameters, which can be computationally intensive. Additionally, interpretation is more complex due to the regularization terms. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the penalized partial likelihood function, incorporating both L1 (Lasso) and L2 (Ridge) penalties; application of regularization techniques to estimate the regression coefficients by maximizing the penalized log-partial likelihood; performing cross-validation to select optimal values for the penalty parameters (alpha and l1_ratio); and the calculation of the hazard function and survival function for new data using the estimated regularized coefficients.

1. The [cox net survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.linear_model</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">l1_ratio</span> = ElasticNet mixing parameter made to vary between 0.10, 0.50 and 1.00
    * <span style="color: #FF0000">alpha_min_ratio</span> = minimum alpha of the regularization path made to vary between 0.0001 and 0.01
3. Hyperparameter tuning was conducted using the 5-fold cross-validation method repeated 5 times with optimal model performance using the concordance index determined for: 
    * <span style="color: #FF0000">l1_ratio</span> = 0.10
    * <span style="color: #FF0000">alpha_min_ratio</span> = 0.01
4. The cross-validated model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7014
5. The apparent model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7419
6. The independent validation model performance of the final model is summarized as follows:
    * **Concordance Index** = 0.7299
7. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of moderate model overfitting.
8. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
9. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.
    


```python
##################################
# Performing hyperparameter tuning
# through K-fold cross-validation
# using the Cox Proportional Hazards Regression Model
##################################
coxns_grid_search.fit(X_train, y_train_array)
```

    Fitting 25 folds for each of 6 candidates, totalling 150 fits
    




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;coxns&#x27;, CoxnetSurvivalAnalysis())]),
             n_jobs=-1,
             param_grid={&#x27;coxns__alpha_min_ratio&#x27;: [0.0001, 0.01],
                         &#x27;coxns__fit_baseline_model&#x27;: [True],
                         &#x27;coxns__l1_ratio&#x27;: [0.1, 0.5, 1.0]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;coxns&#x27;, CoxnetSurvivalAnalysis())]),
             n_jobs=-1,
             param_grid={&#x27;coxns__alpha_min_ratio&#x27;: [0.0001, 0.01],
                         &#x27;coxns__fit_baseline_model&#x27;: [True],
                         &#x27;coxns__l1_ratio&#x27;: [0.1, 0.5, 1.0]},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxns&#x27;,
                 CoxnetSurvivalAnalysis(alpha_min_ratio=0.01,
                                        fit_baseline_model=True,
                                        l1_ratio=0.1))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CoxnetSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, fit_baseline_model=True,
                       l1_ratio=0.1)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Summarizing the hyperparameter tuning 
# results from K-fold cross-validation
##################################
coxns_grid_search_results = pd.DataFrame(coxns_grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
coxns_grid_search_results.loc[:, ~coxns_grid_search_results.columns.str.endswith('_time')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_coxns__alpha_min_ratio</th>
      <th>param_coxns__fit_baseline_model</th>
      <th>param_coxns__l1_ratio</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>...</th>
      <th>split18_test_score</th>
      <th>split19_test_score</th>
      <th>split20_test_score</th>
      <th>split21_test_score</th>
      <th>split22_test_score</th>
      <th>split23_test_score</th>
      <th>split24_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>True</td>
      <td>0.1</td>
      <td>{'coxns__alpha_min_ratio': 0.01, 'coxns__fit_b...</td>
      <td>0.761468</td>
      <td>0.705882</td>
      <td>0.785714</td>
      <td>0.844444</td>
      <td>0.514286</td>
      <td>0.601227</td>
      <td>...</td>
      <td>0.630</td>
      <td>0.675472</td>
      <td>0.727679</td>
      <td>0.683962</td>
      <td>0.694823</td>
      <td>0.525641</td>
      <td>0.796875</td>
      <td>0.701369</td>
      <td>0.085525</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>True</td>
      <td>0.1</td>
      <td>{'coxns__alpha_min_ratio': 0.0001, 'coxns__fit...</td>
      <td>0.761468</td>
      <td>0.702786</td>
      <td>0.790816</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.619632</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.675472</td>
      <td>0.723214</td>
      <td>0.669811</td>
      <td>0.694823</td>
      <td>0.521368</td>
      <td>0.802083</td>
      <td>0.701345</td>
      <td>0.086059</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0001</td>
      <td>True</td>
      <td>0.5</td>
      <td>{'coxns__alpha_min_ratio': 0.0001, 'coxns__fit...</td>
      <td>0.761468</td>
      <td>0.702786</td>
      <td>0.785714</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.619632</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.675472</td>
      <td>0.718750</td>
      <td>0.665094</td>
      <td>0.694823</td>
      <td>0.525641</td>
      <td>0.802083</td>
      <td>0.701266</td>
      <td>0.086197</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0001</td>
      <td>True</td>
      <td>1.0</td>
      <td>{'coxns__alpha_min_ratio': 0.0001, 'coxns__fit...</td>
      <td>0.761468</td>
      <td>0.702786</td>
      <td>0.785714</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.619632</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.675472</td>
      <td>0.718750</td>
      <td>0.665094</td>
      <td>0.694823</td>
      <td>0.525641</td>
      <td>0.796875</td>
      <td>0.700668</td>
      <td>0.086233</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0100</td>
      <td>True</td>
      <td>0.5</td>
      <td>{'coxns__alpha_min_ratio': 0.01, 'coxns__fit_b...</td>
      <td>0.758410</td>
      <td>0.708978</td>
      <td>0.790816</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.619632</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.675472</td>
      <td>0.718750</td>
      <td>0.669811</td>
      <td>0.694823</td>
      <td>0.517094</td>
      <td>0.802083</td>
      <td>0.700332</td>
      <td>0.086584</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0100</td>
      <td>True</td>
      <td>1.0</td>
      <td>{'coxns__alpha_min_ratio': 0.01, 'coxns__fit_b...</td>
      <td>0.758410</td>
      <td>0.705882</td>
      <td>0.790816</td>
      <td>0.848889</td>
      <td>0.492857</td>
      <td>0.616564</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.675472</td>
      <td>0.714286</td>
      <td>0.669811</td>
      <td>0.694823</td>
      <td>0.517094</td>
      <td>0.802083</td>
      <td>0.700281</td>
      <td>0.086774</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 32 columns</p>
</div>




```python
##################################
# Identifying the best model
##################################
coxns_best_model_train_cv = coxns_grid_search.best_estimator_
print('Best Cox Proportional Hazards Regression Model using the Cross-Validated Train Data: ')
print(f"Best Model Parameters: {coxns_grid_search.best_params_}")
```

    Best Cox Proportional Hazards Regression Model using the Cross-Validated Train Data: 
    Best Model Parameters: {'coxns__alpha_min_ratio': 0.01, 'coxns__fit_baseline_model': True, 'coxns__l1_ratio': 0.1}
    


```python
##################################
# Obtaining the cross-validation model performance of the 
# optimal Cox Net Survival Model
# on the train set
##################################
optimal_coxns_heart_failure_y_crossvalidation_ci = coxns_grid_search.best_score_
print(f"Cross-Validation Concordance Index: {optimal_coxns_heart_failure_y_crossvalidation_ci}")
```

    Cross-Validation Concordance Index: 0.7013694603679497
    


```python
##################################
# Formulating a Cox Net Survival Model
# with optimal hyperparameters
##################################
optimal_coxns_model = coxns_grid_search.best_estimator_
optimal_coxns_model.fit(X_train, y_train_array)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxns&#x27;,
                 CoxnetSurvivalAnalysis(alpha_min_ratio=0.01,
                                        fit_baseline_model=True,
                                        l1_ratio=0.1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;coxns&#x27;,
                 CoxnetSurvivalAnalysis(alpha_min_ratio=0.01,
                                        fit_baseline_model=True,
                                        l1_ratio=0.1))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CoxnetSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, fit_baseline_model=True,
                       l1_ratio=0.1)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Measuring model performance of the 
# optimal Cox Net Survival Model
# on the train set
##################################
optimal_coxns_heart_failure_y_train_pred = optimal_coxns_model.predict(X_train)
optimal_coxns_heart_failure_y_train_ci = concordance_index_censored(y_train_array['DEATH_EVENT'], 
                                                                    y_train_array['TIME'], 
                                                                    optimal_coxns_heart_failure_y_train_pred)[0]
print(f"Apparent Concordance Index: {optimal_coxns_heart_failure_y_train_ci}")
```

    Apparent Concordance Index: 0.7419406319821258
    


```python
##################################
# Measuring model performance of the 
# optimal Cox Net Survival Model
# on the validation set
##################################
optimal_coxns_heart_failure_y_validation_pred = optimal_coxns_model.predict(X_validation)
optimal_coxns_heart_failure_y_validation_ci = concordance_index_censored(y_validation_array['DEATH_EVENT'], 
                                                                         y_validation_array['TIME'], 
                                                                         optimal_coxns_heart_failure_y_validation_pred)[0]
print(f"Validation Concordance Index: {optimal_coxns_heart_failure_y_validation_ci}")
```

    Validation Concordance Index: 0.7298772169167803
    


```python
##################################
# Gathering the concordance indices
# from the train and tests sets for 
# Cox Net Survival Model
##################################
coxns_set = pd.DataFrame(["Train","Cross-Validation","Validation"])
coxns_ci_values = pd.DataFrame([optimal_coxns_heart_failure_y_train_ci,
                                optimal_coxns_heart_failure_y_crossvalidation_ci,
                                optimal_coxns_heart_failure_y_validation_ci])
coxns_method = pd.DataFrame(["COXNS"]*3)
coxns_summary = pd.concat([coxns_set, 
                           coxns_ci_values,
                           coxns_method], axis=1)
coxns_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxns_summary.reset_index(inplace=True, drop=True)
display(coxns_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.741941</td>
      <td>COXNS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.701369</td>
      <td>COXNS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>0.729877</td>
      <td>COXNS</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
heart_failure_validation.reset_index(drop=True, inplace=True)
kmf = KaplanMeierFitter()
heart_failure_validation['Predicted_Risks_CoxNS'] = optimal_coxns_heart_failure_y_validation_pred
heart_failure_validation['Predicted_RiskGroups_CoxNS'] = risk_groups = pd.qcut(heart_failure_validation['Predicted_Risks_CoxNS'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = heart_failure_validation[risk_groups == group]
    kmf.fit(group_data['TIME'], event_observed=group_data['DEATH_EVENT'], label=group)
    kmf.plot_survival_function()

plt.title('COXNS Survival Probabilities by Predicted Risk Groups on Validation Set')
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')
plt.show()
```


    
![png](output_181_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
validation_case_details = X_validation.iloc[[5, 10, 15, 20, 25]]
display(validation_case_details)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>60.0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>42.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50.0</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>57.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45.0</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>127.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Gathering the event and duration information
# for 5 test case samples
##################################
print(y_validation_array[[5, 10, 15, 20, 25]])
```

    [(False, 258) ( True,  65) (False,  90) (False,  79) ( True,  14)]
    


```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(heart_failure_validation.loc[[5, 10, 15, 20, 25]][['Predicted_RiskGroups_CoxNS']])
```

       Predicted_RiskGroups_CoxNS
    5                    Low-Risk
    10                  High-Risk
    15                  High-Risk
    20                  High-Risk
    25                  High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 validation cases
##################################
validation_case = X_validation.iloc[[5, 10, 15, 20, 25]]
validation_case_labels = ['Patient_5','Patient_10','Patient_15','Patient_20','Patient_25',]
validation_case_cumulative_hazard_function = optimal_coxns_model.predict_cumulative_hazard_function(validation_case)
validation_case_survival_function = optimal_coxns_model.predict_survival_function(validation_case)

fig, ax = plt.subplots(1,2,figsize=(17, 8))
for hazard_prediction, survival_prediction in zip(validation_case_cumulative_hazard_function, validation_case_survival_function):
    ax[0].step(hazard_prediction.x,hazard_prediction(hazard_prediction.x),where='post')
    ax[1].step(survival_prediction.x,survival_prediction(survival_prediction.x),where='post')
ax[0].set_title('COXNS Cumulative Hazard for 5 Validation Cases')
ax[0].set_xlabel('TIME')
ax[0].set_ylim(0,2)
ax[0].set_ylabel('Cumulative Hazard')
ax[0].legend(validation_case_labels, loc="upper left")
ax[1].set_title('COXNS Survival Function for 5 Validation Cases')
ax[1].set_xlabel('TIME')
ax[1].set_ylabel('DEATH_EVENT Survival Probability')
ax[1].legend(validation_case_labels, loc="lower left")
plt.show()
```


    
![png](output_185_0.png)
    



```python
##################################
# Saving the best Cox Proportional Hazards Regression Model
# developed from the original training data
################################## 
joblib.dump(coxns_best_model_train_cv, 
            os.path.join("..", MODELS_PATH, "coxns_best_model.pkl"))
```




    ['..\\models\\coxns_best_model.pkl']



### 1.6.6 Survival Tree Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.6"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Survival Trees](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476296) are non-parametric models that partition the data into subgroups (nodes) based on the values of predictor variables, creating a tree-like structure. The tree is built by recursively splitting the data at nodes where the differences in survival times between subgroups are maximized. Each terminal node represents a different survival function. The method have no assumptions about the underlying distribution of survival times, can capture interactions between variables naturally and applies an interpretable visual representation. However, the process can be prone to overfitting, especially with small datasets, and may be less accurate compared to ensemble methods like Random Survival Forest. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves recursively splitting the data at nodes to maximize the differences in survival times between subgroups with the splitting criteria often involving statistical tests (e.g., log-rank test); choosing the best predictor variable and split point at each node that maximizes the separation of survival times; continuously splitting until stopping criteria are met (e.g., minimum number of observations in a node, maximum tree depth); and estimating the survival function based on the survival times of the observations at each terminal node.

1. The [survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html) from the <mark style="background-color: #CCECFF"><b>sksurv.tree</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">min_samples_split</span> = minimum number of samples required to split an internal node made to vary between 10, 15 and 20
    * <span style="color: #FF0000">min_samples_leaf</span> = minimum number of samples required to be at a leaf node made to vary between 3 and 6
3. Hyperparameter tuning was conducted using the 5-fold cross-validation method repeated 5 times with optimal model performance using the concordance index determined for: 
    * <span style="color: #FF0000">min_samples_split</span> = 20
    * <span style="color: #FF0000">min_samples_leaf</span> = 6
4. The cross-validated model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.6542
5. The apparent model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7992
6. The independent validation model performance of the final model is summarized as follows:
    * **Concordance Index** = 0.6446
7. Significant difference in the apparent and cross-validated model performance observed, indicative of the presence of excessive model overfitting.
8. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated non-optimal differentiation across the entire duration.
9. Hazard and survival probability estimations for 5 sampled cases demonstrated non-optimal profiles.
    


```python
##################################
# Performing hyperparameter tuning
# through K-fold cross-validation
# using the Survival Tree Model
##################################
stree_grid_search.fit(X_train, y_train_array)
```

    Fitting 25 folds for each of 6 candidates, totalling 150 fits
    




<style>#sk-container-id-5 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-5 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;stree&#x27;, SurvivalTree())]),
             n_jobs=-1,
             param_grid={&#x27;stree__min_samples_leaf&#x27;: [3, 6],
                         &#x27;stree__min_samples_split&#x27;: [10, 15, 20],
                         &#x27;stree__random_state&#x27;: [88888888]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-31" type="checkbox" ><label for="sk-estimator-id-31" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;stree&#x27;, SurvivalTree())]),
             n_jobs=-1,
             param_grid={&#x27;stree__min_samples_leaf&#x27;: [3, 6],
                         &#x27;stree__min_samples_split&#x27;: [10, 15, 20],
                         &#x27;stree__random_state&#x27;: [88888888]},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-32" type="checkbox" ><label for="sk-estimator-id-32" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;stree&#x27;,
                 SurvivalTree(min_samples_leaf=6, min_samples_split=20,
                              random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-33" type="checkbox" ><label for="sk-estimator-id-33" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-34" type="checkbox" ><label for="sk-estimator-id-34" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-35" type="checkbox" ><label for="sk-estimator-id-35" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-36" type="checkbox" ><label for="sk-estimator-id-36" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-37" type="checkbox" ><label for="sk-estimator-id-37" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-38" type="checkbox" ><label for="sk-estimator-id-38" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">SurvivalTree</label><div class="sk-toggleable__content fitted"><pre>SurvivalTree(min_samples_leaf=6, min_samples_split=20, random_state=88888888)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Summarizing the hyperparameter tuning 
# results from K-fold cross-validation
##################################
stree_grid_search_results = pd.DataFrame(stree_grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
stree_grid_search_results.loc[:, ~stree_grid_search_results.columns.str.endswith('_time')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_stree__min_samples_leaf</th>
      <th>param_stree__min_samples_split</th>
      <th>param_stree__random_state</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>...</th>
      <th>split18_test_score</th>
      <th>split19_test_score</th>
      <th>split20_test_score</th>
      <th>split21_test_score</th>
      <th>split22_test_score</th>
      <th>split23_test_score</th>
      <th>split24_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>20</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 6, 'stree__min_sam...</td>
      <td>0.749235</td>
      <td>0.715170</td>
      <td>0.721939</td>
      <td>0.726667</td>
      <td>0.560714</td>
      <td>0.562883</td>
      <td>...</td>
      <td>0.7250</td>
      <td>0.609434</td>
      <td>0.680804</td>
      <td>0.627358</td>
      <td>0.698910</td>
      <td>0.611111</td>
      <td>0.494792</td>
      <td>0.654169</td>
      <td>0.072607</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>20</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 3, 'stree__min_sam...</td>
      <td>0.669725</td>
      <td>0.690402</td>
      <td>0.653061</td>
      <td>0.777778</td>
      <td>0.553571</td>
      <td>0.475460</td>
      <td>...</td>
      <td>0.7250</td>
      <td>0.588679</td>
      <td>0.671875</td>
      <td>0.632075</td>
      <td>0.716621</td>
      <td>0.638889</td>
      <td>0.505208</td>
      <td>0.646178</td>
      <td>0.076711</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>15</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 6, 'stree__min_sam...</td>
      <td>0.743119</td>
      <td>0.687307</td>
      <td>0.798469</td>
      <td>0.726667</td>
      <td>0.539286</td>
      <td>0.553681</td>
      <td>...</td>
      <td>0.6325</td>
      <td>0.658491</td>
      <td>0.629464</td>
      <td>0.627358</td>
      <td>0.660763</td>
      <td>0.621795</td>
      <td>0.486979</td>
      <td>0.636490</td>
      <td>0.079302</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>10</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 6, 'stree__min_sam...</td>
      <td>0.718654</td>
      <td>0.681115</td>
      <td>0.801020</td>
      <td>0.724444</td>
      <td>0.546429</td>
      <td>0.558282</td>
      <td>...</td>
      <td>0.6500</td>
      <td>0.715094</td>
      <td>0.642857</td>
      <td>0.587264</td>
      <td>0.643052</td>
      <td>0.611111</td>
      <td>0.486979</td>
      <td>0.634086</td>
      <td>0.078157</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>15</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 3, 'stree__min_sam...</td>
      <td>0.669725</td>
      <td>0.673375</td>
      <td>0.673469</td>
      <td>0.746667</td>
      <td>0.564286</td>
      <td>0.438650</td>
      <td>...</td>
      <td>0.6425</td>
      <td>0.598113</td>
      <td>0.609375</td>
      <td>0.632075</td>
      <td>0.663488</td>
      <td>0.608974</td>
      <td>0.502604</td>
      <td>0.624111</td>
      <td>0.072077</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>10</td>
      <td>88888888</td>
      <td>{'stree__min_samples_leaf': 3, 'stree__min_sam...</td>
      <td>0.646789</td>
      <td>0.684211</td>
      <td>0.678571</td>
      <td>0.691111</td>
      <td>0.582143</td>
      <td>0.435583</td>
      <td>...</td>
      <td>0.6300</td>
      <td>0.635849</td>
      <td>0.587054</td>
      <td>0.509434</td>
      <td>0.632153</td>
      <td>0.647436</td>
      <td>0.489583</td>
      <td>0.609106</td>
      <td>0.074834</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 32 columns</p>
</div>




```python
##################################
# Identifying the best model
##################################
stree_best_model_train_cv = stree_grid_search.best_estimator_
print('Best Survival Tree Model using the Cross-Validated Train Data: ')
print(f"Best Model Parameters: {stree_grid_search.best_params_}")
```

    Best Survival Tree Model using the Cross-Validated Train Data: 
    Best Model Parameters: {'stree__min_samples_leaf': 6, 'stree__min_samples_split': 20, 'stree__random_state': 88888888}
    


```python
##################################
# Obtaining the cross-validation model performance of the 
# optimal Survival Tree Model
# on the train set
##################################
optimal_stree_heart_failure_y_crossvalidation_ci = stree_grid_search.best_score_
print(f"Cross-Validation Concordance Index: {optimal_stree_heart_failure_y_crossvalidation_ci}")
```

    Cross-Validation Concordance Index: 0.6541686643258245
    


```python
##################################
# Formulating a Survival Tree Model
# with optimal hyperparameters
##################################
optimal_stree_model = stree_grid_search.best_estimator_
optimal_stree_model.fit(X_train, y_train_array)
```




<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;stree&#x27;,
                 SurvivalTree(min_samples_leaf=6, min_samples_split=20,
                              random_state=88888888))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-39" type="checkbox" ><label for="sk-estimator-id-39" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;stree&#x27;,
                 SurvivalTree(min_samples_leaf=6, min_samples_split=20,
                              random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-40" type="checkbox" ><label for="sk-estimator-id-40" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-41" type="checkbox" ><label for="sk-estimator-id-41" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-42" type="checkbox" ><label for="sk-estimator-id-42" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-43" type="checkbox" ><label for="sk-estimator-id-43" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-44" type="checkbox" ><label for="sk-estimator-id-44" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-45" type="checkbox" ><label for="sk-estimator-id-45" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">SurvivalTree</label><div class="sk-toggleable__content fitted"><pre>SurvivalTree(min_samples_leaf=6, min_samples_split=20, random_state=88888888)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Measuring model performance of the 
# optimal Survival Tree Model
# on the train set
##################################
optimal_stree_heart_failure_y_train_pred = optimal_stree_model.predict(X_train)
optimal_stree_heart_failure_y_train_ci = concordance_index_censored(y_train_array['DEATH_EVENT'], 
                                                                    y_train_array['TIME'], 
                                                                    optimal_stree_heart_failure_y_train_pred)[0]
print(f"Apparent Concordance Index: {optimal_stree_heart_failure_y_train_ci}")
```

    Apparent Concordance Index: 0.7992339610596872
    


```python
##################################
# Measuring model performance of the 
# optimal Survival Tree Model
# on the validation set
##################################
optimal_stree_heart_failure_y_validation_pred = optimal_stree_model.predict(X_validation)
optimal_stree_heart_failure_y_validation_ci = concordance_index_censored(y_validation_array['DEATH_EVENT'], 
                                                                         y_validation_array['TIME'], 
                                                                         optimal_stree_heart_failure_y_validation_pred)[0]
print(f"Validation Concordance Index: {optimal_stree_heart_failure_y_validation_ci}")
```

    Validation Concordance Index: 0.6446111869031378
    


```python
##################################
# Gathering the concordance indices
# from the train and tests sets for 
# Survival Tree Model
##################################
stree_set = pd.DataFrame(["Train","Cross-Validation","Validation"])
stree_ci_values = pd.DataFrame([optimal_stree_heart_failure_y_train_ci,
                                optimal_stree_heart_failure_y_crossvalidation_ci,
                                optimal_stree_heart_failure_y_validation_ci])
stree_method = pd.DataFrame(["STREE"]*3)
stree_summary = pd.concat([stree_set, 
                           stree_ci_values,
                           stree_method], axis=1)
stree_summary.columns = ['Set', 'Concordance.Index', 'Method']
stree_summary.reset_index(inplace=True, drop=True)
display(stree_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.799234</td>
      <td>STREE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.654169</td>
      <td>STREE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>0.644611</td>
      <td>STREE</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
heart_failure_validation.reset_index(drop=True, inplace=True)
kmf = KaplanMeierFitter()
heart_failure_validation['Predicted_Risks_STree'] = optimal_stree_heart_failure_y_validation_pred
heart_failure_validation['Predicted_RiskGroups_STree'] = risk_groups = pd.qcut(heart_failure_validation['Predicted_Risks_STree'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = heart_failure_validation[risk_groups == group]
    kmf.fit(group_data['TIME'], event_observed=group_data['DEATH_EVENT'], label=group)
    kmf.plot_survival_function()

plt.title('STREE Survival Probabilities by Predicted Risk Groups on Validation Set')
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')
plt.show()
```


    
![png](output_196_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
validation_case_details = X_validation.iloc[[5, 10, 15, 20, 25]]
display(validation_case_details)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>60.0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>42.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50.0</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>57.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45.0</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>127.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Gathering the event and duration information
# for 5 test case samples
##################################
print(y_validation_array[[5, 10, 15, 20, 25]])
```

    [(False, 258) ( True,  65) (False,  90) (False,  79) ( True,  14)]
    


```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(heart_failure_validation.loc[[5, 10, 15, 20, 25]][['Predicted_RiskGroups_STree']])
```

       Predicted_RiskGroups_STree
    5                    Low-Risk
    10                  High-Risk
    15                  High-Risk
    20                  High-Risk
    25                  High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 validation cases
##################################
validation_case = X_validation.iloc[[5, 10, 15, 20, 25]]
validation_case_labels = ['Patient_5','Patient_10','Patient_15','Patient_20','Patient_25',]
validation_case_cumulative_hazard_function = optimal_stree_model.predict_cumulative_hazard_function(validation_case)
validation_case_survival_function = optimal_stree_model.predict_survival_function(validation_case)

fig, ax = plt.subplots(1,2,figsize=(17, 8))
for hazard_prediction, survival_prediction in zip(validation_case_cumulative_hazard_function, validation_case_survival_function):
    ax[0].step(hazard_prediction.x,hazard_prediction(hazard_prediction.x),where='post')
    ax[1].step(survival_prediction.x,survival_prediction(survival_prediction.x),where='post')
ax[0].set_title('STREE Cumulative Hazard for 5 Validation Cases')
ax[0].set_xlabel('TIME')
ax[0].set_ylim(0,2)
ax[0].set_ylabel('Cumulative Hazard')
ax[0].legend(validation_case_labels, loc="upper left")
ax[1].set_title('STREE Survival Function for 5 Validation Cases')
ax[1].set_xlabel('TIME')
ax[1].set_ylabel('DEATH_EVENT Survival Probability')
ax[1].legend(validation_case_labels, loc="lower left")
plt.show()
```


    
![png](output_200_0.png)
    



```python
##################################
# Saving the best Survival Tree Model
# developed from the original training data
################################## 
joblib.dump(stree_best_model_train_cv, 
            os.path.join("..", MODELS_PATH, "stree_best_model.pkl"))
```




    ['..\\models\\stree_best_model.pkl']



### 1.6.7 Random Survival Forest Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.7"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Random Survival Forest](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Random-survival-forests/10.1214/08-AOAS169.full) is an ensemble method that builds multiple survival trees and averages their predictions. The model combines the predictions of multiple survival trees, each built on a bootstrap sample of the data and a random subset of predictors. It uses the concept of ensemble learning to improve predictive accuracy and robustness. As a method, it handles high-dimensional data and complex interactions between variables well; can be more accurate and robust than a single survival tree; and provides measures of variable importance. However, the process can be bomputationally intensive due to the need to build multiple trees, and may be less interpretable than single trees or parametric models like the Cox model. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves generating multiple bootstrap samples from the original dataset; building a survival tree by recursively splitting the data at nodes using a random subset of predictor variables for each bootstrap sample; combining the predictions of all survival trees to form the random survival forest and averaging the survival functions predicted by all trees in the forest to obtain the final survival function for new data.

1. The [random survival forest model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html) from the <mark style="background-color: #CCECFF"><b>sksurv.ensemble</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">n_estimators</span> = number of trees in the forest made to vary between 100, 200 and 300
    * <span style="color: #FF0000">min_samples_split</span> = minimum number of samples required to split an internal node made to vary between 10, 15 and 20
3. Hyperparameter tuning was conducted using the 5-fold cross-validation method repeated 5 times with optimal model performance using the concordance index determined for: 
    * <span style="color: #FF0000">n_estimators</span> = 300
    * <span style="color: #FF0000">min_samples_split</span> = 10
4. The cross-validated model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.7091
5. The apparent model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.8714
6. The independent test model performance of the final model is summarized as follows:
    * **Concordance Index** = 0.6930
7. Significant difference in the apparent and cross-validated model performance observed, indicative of the presence of excessive model overfitting.
8. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
9. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Performing hyperparameter tuning
# through K-fold cross-validation
# using the Random Survival Forest Model
##################################
rsf_grid_search.fit(X_train, y_train_array)
```

    Fitting 25 folds for each of 9 candidates, totalling 225 fits
    




<style>#sk-container-id-7 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-7 {
  color: var(--sklearn-color-text);
}

#sk-container-id-7 pre {
  padding: 0;
}

#sk-container-id-7 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-7 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-7 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-7 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-7 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-7 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-7 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-7 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-7 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-7 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-7 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-7 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-7 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-7 div.sk-label label.sk-toggleable__label,
#sk-container-id-7 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-7 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-7 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-7 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-7 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-7 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-7 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-7 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-7 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-7 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;rsf&#x27;, RandomSurvivalForest())]),
             n_jobs=-1,
             param_grid={&#x27;rsf__min_samples_split&#x27;: [10, 15, 20],
                         &#x27;rsf__n_estimators&#x27;: [100, 200, 300],
                         &#x27;rsf__random_state&#x27;: [88888888]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-46" type="checkbox" ><label for="sk-estimator-id-46" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;rsf&#x27;, RandomSurvivalForest())]),
             n_jobs=-1,
             param_grid={&#x27;rsf__min_samples_split&#x27;: [10, 15, 20],
                         &#x27;rsf__n_estimators&#x27;: [100, 200, 300],
                         &#x27;rsf__random_state&#x27;: [88888888]},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-47" type="checkbox" ><label for="sk-estimator-id-47" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;rsf&#x27;,
                 RandomSurvivalForest(min_samples_split=10, n_estimators=300,
                                      random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-48" type="checkbox" ><label for="sk-estimator-id-48" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-49" type="checkbox" ><label for="sk-estimator-id-49" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-50" type="checkbox" ><label for="sk-estimator-id-50" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-51" type="checkbox" ><label for="sk-estimator-id-51" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-52" type="checkbox" ><label for="sk-estimator-id-52" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-53" type="checkbox" ><label for="sk-estimator-id-53" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">RandomSurvivalForest</label><div class="sk-toggleable__content fitted"><pre>RandomSurvivalForest(min_samples_split=10, n_estimators=300,
                     random_state=88888888)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Summarizing the hyperparameter tuning 
# results from K-fold cross-validation
##################################
rsf_grid_search_results = pd.DataFrame(rsf_grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
rsf_grid_search_results.loc[:, ~rsf_grid_search_results.columns.str.endswith('_time')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_rsf__min_samples_split</th>
      <th>param_rsf__n_estimators</th>
      <th>param_rsf__random_state</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>...</th>
      <th>split18_test_score</th>
      <th>split19_test_score</th>
      <th>split20_test_score</th>
      <th>split21_test_score</th>
      <th>split22_test_score</th>
      <th>split23_test_score</th>
      <th>split24_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 10, 'rsf__n_estimat...</td>
      <td>0.737003</td>
      <td>0.687307</td>
      <td>0.744898</td>
      <td>0.840000</td>
      <td>0.585714</td>
      <td>0.650307</td>
      <td>...</td>
      <td>0.660</td>
      <td>0.701887</td>
      <td>0.772321</td>
      <td>0.759434</td>
      <td>0.727520</td>
      <td>0.598291</td>
      <td>0.671875</td>
      <td>0.709097</td>
      <td>0.068130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 20, 'rsf__n_estimat...</td>
      <td>0.730887</td>
      <td>0.705882</td>
      <td>0.734694</td>
      <td>0.831111</td>
      <td>0.592857</td>
      <td>0.647239</td>
      <td>...</td>
      <td>0.675</td>
      <td>0.698113</td>
      <td>0.758929</td>
      <td>0.754717</td>
      <td>0.741144</td>
      <td>0.598291</td>
      <td>0.671875</td>
      <td>0.707597</td>
      <td>0.066923</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 10, 'rsf__n_estimat...</td>
      <td>0.755352</td>
      <td>0.690402</td>
      <td>0.755102</td>
      <td>0.844444</td>
      <td>0.578571</td>
      <td>0.647239</td>
      <td>...</td>
      <td>0.665</td>
      <td>0.713208</td>
      <td>0.772321</td>
      <td>0.754717</td>
      <td>0.727520</td>
      <td>0.581197</td>
      <td>0.640625</td>
      <td>0.707268</td>
      <td>0.074312</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 10, 'rsf__n_estimat...</td>
      <td>0.743119</td>
      <td>0.684211</td>
      <td>0.739796</td>
      <td>0.835556</td>
      <td>0.592857</td>
      <td>0.644172</td>
      <td>...</td>
      <td>0.660</td>
      <td>0.705660</td>
      <td>0.758929</td>
      <td>0.754717</td>
      <td>0.732970</td>
      <td>0.594017</td>
      <td>0.651042</td>
      <td>0.707263</td>
      <td>0.068872</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 15, 'rsf__n_estimat...</td>
      <td>0.733945</td>
      <td>0.708978</td>
      <td>0.739796</td>
      <td>0.831111</td>
      <td>0.578571</td>
      <td>0.647239</td>
      <td>...</td>
      <td>0.675</td>
      <td>0.701887</td>
      <td>0.754464</td>
      <td>0.754717</td>
      <td>0.741144</td>
      <td>0.594017</td>
      <td>0.666667</td>
      <td>0.706527</td>
      <td>0.068454</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 20, 'rsf__n_estimat...</td>
      <td>0.730887</td>
      <td>0.702786</td>
      <td>0.734694</td>
      <td>0.835556</td>
      <td>0.578571</td>
      <td>0.653374</td>
      <td>...</td>
      <td>0.675</td>
      <td>0.683019</td>
      <td>0.758929</td>
      <td>0.759434</td>
      <td>0.732970</td>
      <td>0.598291</td>
      <td>0.666667</td>
      <td>0.706351</td>
      <td>0.067533</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 20, 'rsf__n_estimat...</td>
      <td>0.740061</td>
      <td>0.699690</td>
      <td>0.739796</td>
      <td>0.840000</td>
      <td>0.557143</td>
      <td>0.650307</td>
      <td>...</td>
      <td>0.670</td>
      <td>0.698113</td>
      <td>0.772321</td>
      <td>0.745283</td>
      <td>0.722071</td>
      <td>0.594017</td>
      <td>0.671875</td>
      <td>0.706127</td>
      <td>0.069620</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 15, 'rsf__n_estimat...</td>
      <td>0.737003</td>
      <td>0.690402</td>
      <td>0.739796</td>
      <td>0.835556</td>
      <td>0.571429</td>
      <td>0.653374</td>
      <td>...</td>
      <td>0.675</td>
      <td>0.690566</td>
      <td>0.758929</td>
      <td>0.768868</td>
      <td>0.735695</td>
      <td>0.598291</td>
      <td>0.666667</td>
      <td>0.704593</td>
      <td>0.071233</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'rsf__min_samples_split': 15, 'rsf__n_estimat...</td>
      <td>0.740061</td>
      <td>0.684211</td>
      <td>0.755102</td>
      <td>0.840000</td>
      <td>0.571429</td>
      <td>0.644172</td>
      <td>...</td>
      <td>0.665</td>
      <td>0.698113</td>
      <td>0.745536</td>
      <td>0.745283</td>
      <td>0.732970</td>
      <td>0.585470</td>
      <td>0.661458</td>
      <td>0.703792</td>
      <td>0.074270</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>9 rows × 32 columns</p>
</div>




```python
##################################
# Identifying the best model
##################################
rsf_best_model_train_cv = rsf_grid_search.best_estimator_
print('Best Random Survival Forest Model using the Cross-Validated Train Data: ')
print(f"Best Model Parameters: {rsf_grid_search.best_params_}")
```

    Best Random Survival Forest Model using the Cross-Validated Train Data: 
    Best Model Parameters: {'rsf__min_samples_split': 10, 'rsf__n_estimators': 300, 'rsf__random_state': 88888888}
    


```python
##################################
# Obtaining the cross-validation model performance of the 
# optimal Random Survival Forest Model
# on the train set
##################################
optimal_rsf_heart_failure_y_crossvalidation_ci = rsf_grid_search.best_score_
print(f"Cross-Validation Concordance Index: {optimal_rsf_heart_failure_y_crossvalidation_ci}")
```

    Cross-Validation Concordance Index: 0.7090965292195327
    


```python
##################################
# Formulating a Random Survival Forest Model
# with optimal hyperparameters
##################################
optimal_rsf_model = rsf_grid_search.best_estimator_
optimal_rsf_model.fit(X_train, y_train_array)
```




<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;rsf&#x27;,
                 RandomSurvivalForest(min_samples_split=10, n_estimators=300,
                                      random_state=88888888))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-54" type="checkbox" ><label for="sk-estimator-id-54" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;rsf&#x27;,
                 RandomSurvivalForest(min_samples_split=10, n_estimators=300,
                                      random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-55" type="checkbox" ><label for="sk-estimator-id-55" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-56" type="checkbox" ><label for="sk-estimator-id-56" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-57" type="checkbox" ><label for="sk-estimator-id-57" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-58" type="checkbox" ><label for="sk-estimator-id-58" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-59" type="checkbox" ><label for="sk-estimator-id-59" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-60" type="checkbox" ><label for="sk-estimator-id-60" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">RandomSurvivalForest</label><div class="sk-toggleable__content fitted"><pre>RandomSurvivalForest(min_samples_split=10, n_estimators=300,
                     random_state=88888888)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Measuring model performance of the 
# optimal Random Survival Forest Model
# on the train set
##################################
optimal_rsf_heart_failure_y_train_pred = optimal_rsf_model.predict(X_train)
optimal_rsf_heart_failure_y_train_ci = concordance_index_censored(y_train_array['DEATH_EVENT'],
                                                                  y_train_array['TIME'], 
                                                                  optimal_rsf_heart_failure_y_train_pred)[0]
print(f"Apparent Concordance Index: {optimal_rsf_heart_failure_y_train_ci}")
```

    Apparent Concordance Index: 0.8713692946058091
    


```python
##################################
# Measuring model performance of the 
# optimal Random Survival Forest Model
# on the validation set
##################################
optimal_rsf_heart_failure_y_validation_pred = optimal_rsf_model.predict(X_validation)
optimal_rsf_heart_failure_y_validation_ci = concordance_index_censored(y_validation_array['DEATH_EVENT'], 
                                                                       y_validation_array['TIME'], 
                                                                       optimal_rsf_heart_failure_y_validation_pred)[0]
print(f"Validation Concordance Index: {optimal_rsf_heart_failure_y_validation_ci}")
```

    Validation Concordance Index: 0.6930422919508867
    


```python
##################################
# Gathering the concordance indices
# from the train and tests sets for 
# Random Survival Forest Model
##################################
rsf_set = pd.DataFrame(["Train","Cross-Validation","Validation"])
rsf_ci_values = pd.DataFrame([optimal_rsf_heart_failure_y_train_ci,
                              optimal_rsf_heart_failure_y_crossvalidation_ci,
                              optimal_rsf_heart_failure_y_validation_ci])
rsf_method = pd.DataFrame(["RSF"]*3)
rsf_summary = pd.concat([rsf_set, 
                         rsf_ci_values,
                         rsf_method], axis=1)
rsf_summary.columns = ['Set', 'Concordance.Index', 'Method']
rsf_summary.reset_index(inplace=True, drop=True)
display(rsf_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.871369</td>
      <td>RSF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.709097</td>
      <td>RSF</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>0.693042</td>
      <td>RSF</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
heart_failure_validation.reset_index(drop=True, inplace=True)
kmf = KaplanMeierFitter()
heart_failure_validation['Predicted_Risks_RSF'] = optimal_rsf_heart_failure_y_validation_pred
heart_failure_validation['Predicted_RiskGroups_RSF'] = risk_groups = pd.qcut(heart_failure_validation['Predicted_Risks_RSF'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = heart_failure_validation[risk_groups == group]
    kmf.fit(group_data['TIME'], event_observed=group_data['DEATH_EVENT'], label=group)
    kmf.plot_survival_function()

plt.title('RSF Survival Probabilities by Predicted Risk Groups on Validation Set')
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')
plt.show()
```


    
![png](output_211_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
validation_case_details = X_validation.iloc[[5, 10, 15, 20, 25]]
display(validation_case_details)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>60.0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>42.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50.0</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>57.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45.0</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>127.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Gathering the event and duration information
# for 5 test case samples
##################################
print(y_validation_array[[5, 10, 15, 20, 25]])
```

    [(False, 258) ( True,  65) (False,  90) (False,  79) ( True,  14)]
    


```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(heart_failure_validation.loc[[5, 10, 15, 20, 25]][['Predicted_RiskGroups_RSF']])
```

       Predicted_RiskGroups_RSF
    5                  Low-Risk
    10                High-Risk
    15                High-Risk
    20                High-Risk
    25                High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 validation cases
##################################
validation_case = X_validation.iloc[[5, 10, 15, 20, 25]]
validation_case_labels = ['Patient_5','Patient_10','Patient_15','Patient_20','Patient_25',]
validation_case_cumulative_hazard_function = optimal_rsf_model.predict_cumulative_hazard_function(validation_case)
validation_case_survival_function = optimal_rsf_model.predict_survival_function(validation_case)

fig, ax = plt.subplots(1,2,figsize=(17, 8))
for hazard_prediction, survival_prediction in zip(validation_case_cumulative_hazard_function, validation_case_survival_function):
    ax[0].step(hazard_prediction.x,hazard_prediction(hazard_prediction.x),where='post')
    ax[1].step(survival_prediction.x,survival_prediction(survival_prediction.x),where='post')
ax[0].set_title('RSF Cumulative Hazard for 5 Validation Cases')
ax[0].set_xlabel('TIME')
ax[0].set_ylim(0,2)
ax[0].set_ylabel('Cumulative Hazard')
ax[0].legend(validation_case_labels, loc="upper left")
ax[1].set_title('RSF Survival Function for 5 Validation Cases')
ax[1].set_xlabel('TIME')
ax[1].set_ylabel('DEATH_EVENT Survival Probability')
ax[1].legend(validation_case_labels, loc="lower left")
plt.show()
```


    
![png](output_215_0.png)
    



```python
##################################
# Saving the best Random Survival Forest Model
# developed from the original training data
################################## 
joblib.dump(rsf_best_model_train_cv, 
            os.path.join("..", MODELS_PATH, "rsf_best_model.pkl"))
```




    ['..\\models\\rsf_best_model.pkl']



### 1.6.8 Gradient Boosted Survival Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.8"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Gradient Boosted Survival](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) is an ensemble technique that builds a series of survival trees, where each tree tries to correct the errors of the previous one. The model uses boosting, a sequential technique where each new tree is fit to the residuals of the combined previous trees, and combines the predictions of all the trees to produce a final prediction. As a method, it has high predictive accuracy, the ability to model complex relationships, and reduces bias and variance compared to single-tree models. However, the process can even be more computationally intensive than Random Survival Forest, requires careful tuning of multiple hyperparameters, and makes interpretation challenging due to the complex nature of the model. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves starting with an initial prediction (often the median survival time or a simple model); calculating the residuals (errors) of the current model's predictions; fitting a survival tree to the residuals to learn the errors made by the current model; updating the current model by adding the new tree weighted by a learning rate parameter; repeating previous steps for a fixed number of iterations or until convergence; and summing the predictions of all trees in the sequence to obtain the final survival function for new data.

1. The [gradient boosted survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) from the <mark style="background-color: #CCECFF"><b>sksurv.ensemble</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">n_estimators</span> = number of regression trees to create made to vary between 100, 200 and 300
    * <span style="color: #FF0000">learning_rate</span> = shrinkage parameter for the contribution of each tree made to vary between 0.05, 0.10 and 0.15
3. Hyperparameter tuning was conducted using the 5-fold cross-validation method repeated 5 times with optimal model performance using the concordance index determined for: 
    * <span style="color: #FF0000">n_estimators</span> = 200
    * <span style="color: #FF0000">learning_rate</span> = 0.10
4. The cross-validated model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.6765
5. The apparent model performance of the optimal model is summarized as follows:
    * **Concordance Index** = 0.9275
6. The independent test model performance of the final model is summarized as follows:
    * **Concordance Index** = 0.6575
7. Significant difference in the apparent and cross-validated model performance observed, indicative of the presence of excessive model overfitting.
8. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
9. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.
    


```python
##################################
# Performing hyperparameter tuning
# through K-fold cross-validation
# using the Gradient Boosted Survival Model
##################################
gbs_grid_search.fit(X_train, y_train_array)
```

    Fitting 25 folds for each of 9 candidates, totalling 225 fits
    




<style>#sk-container-id-9 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-9 {
  color: var(--sklearn-color-text);
}

#sk-container-id-9 pre {
  padding: 0;
}

#sk-container-id-9 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-9 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-9 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-9 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-9 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-9 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-9 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-9 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-9 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-9 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-9 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-9 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-9 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-9 div.sk-label label.sk-toggleable__label,
#sk-container-id-9 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-9 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-9 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-9 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-9 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-9 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-9 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-9 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-9 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-9 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;gbs&#x27;,
                                        GradientBoostingSurvivalAnalysis())]),
             n_jobs=-1,
             param_grid={&#x27;gbs__learning_rate&#x27;: [0.05, 0.1, 0.15],
                         &#x27;gbs__n_estimators&#x27;: [100, 200, 300],
                         &#x27;gbs__random_state&#x27;: [88888888]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-61" type="checkbox" ><label for="sk-estimator-id-61" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;numeric_predictors&#x27;,
                                                                         PowerTransformer(),
                                                                         [&#x27;AGE&#x27;,
                                                                          &#x27;EJECTION_FRACTION&#x27;,
                                                                          &#x27;SERUM_CREATININE&#x27;,
                                                                          &#x27;SERUM_SODIUM&#x27;])])),
                                       (&#x27;gbs&#x27;,
                                        GradientBoostingSurvivalAnalysis())]),
             n_jobs=-1,
             param_grid={&#x27;gbs__learning_rate&#x27;: [0.05, 0.1, 0.15],
                         &#x27;gbs__n_estimators&#x27;: [100, 200, 300],
                         &#x27;gbs__random_state&#x27;: [88888888]},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-62" type="checkbox" ><label for="sk-estimator-id-62" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;gbs&#x27;,
                 GradientBoostingSurvivalAnalysis(n_estimators=200,
                                                  random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-63" type="checkbox" ><label for="sk-estimator-id-63" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-64" type="checkbox" ><label for="sk-estimator-id-64" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-65" type="checkbox" ><label for="sk-estimator-id-65" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-66" type="checkbox" ><label for="sk-estimator-id-66" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-67" type="checkbox" ><label for="sk-estimator-id-67" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-68" type="checkbox" ><label for="sk-estimator-id-68" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">GradientBoostingSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>GradientBoostingSurvivalAnalysis(n_estimators=200, random_state=88888888)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Summarizing the hyperparameter tuning 
# results from K-fold cross-validation
##################################
gbs_grid_search_results = pd.DataFrame(gbs_grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
gbs_grid_search_results.loc[:, ~gbs_grid_search_results.columns.str.endswith('_time')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_gbs__learning_rate</th>
      <th>param_gbs__n_estimators</th>
      <th>param_gbs__random_state</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>...</th>
      <th>split18_test_score</th>
      <th>split19_test_score</th>
      <th>split20_test_score</th>
      <th>split21_test_score</th>
      <th>split22_test_score</th>
      <th>split23_test_score</th>
      <th>split24_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>0.10</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.1, 'gbs__n_estimators...</td>
      <td>0.694190</td>
      <td>0.712074</td>
      <td>0.704082</td>
      <td>0.782222</td>
      <td>0.564286</td>
      <td>0.604294</td>
      <td>...</td>
      <td>0.620</td>
      <td>0.656604</td>
      <td>0.736607</td>
      <td>0.712264</td>
      <td>0.697548</td>
      <td>0.594017</td>
      <td>0.692708</td>
      <td>0.676537</td>
      <td>0.059798</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.15</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.15, 'gbs__n_estimator...</td>
      <td>0.700306</td>
      <td>0.718266</td>
      <td>0.704082</td>
      <td>0.760000</td>
      <td>0.571429</td>
      <td>0.592025</td>
      <td>...</td>
      <td>0.630</td>
      <td>0.679245</td>
      <td>0.709821</td>
      <td>0.721698</td>
      <td>0.711172</td>
      <td>0.594017</td>
      <td>0.677083</td>
      <td>0.675917</td>
      <td>0.055144</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.10</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.1, 'gbs__n_estimators...</td>
      <td>0.678899</td>
      <td>0.705882</td>
      <td>0.698980</td>
      <td>0.773333</td>
      <td>0.564286</td>
      <td>0.604294</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.656604</td>
      <td>0.723214</td>
      <td>0.716981</td>
      <td>0.697548</td>
      <td>0.594017</td>
      <td>0.708333</td>
      <td>0.675460</td>
      <td>0.054978</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.05, 'gbs__n_estimator...</td>
      <td>0.712538</td>
      <td>0.708978</td>
      <td>0.688776</td>
      <td>0.788889</td>
      <td>0.550000</td>
      <td>0.598160</td>
      <td>...</td>
      <td>0.615</td>
      <td>0.664151</td>
      <td>0.709821</td>
      <td>0.707547</td>
      <td>0.700272</td>
      <td>0.606838</td>
      <td>0.700521</td>
      <td>0.675307</td>
      <td>0.059465</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.15</td>
      <td>300</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.15, 'gbs__n_estimator...</td>
      <td>0.703364</td>
      <td>0.705882</td>
      <td>0.714286</td>
      <td>0.742222</td>
      <td>0.557143</td>
      <td>0.579755</td>
      <td>...</td>
      <td>0.630</td>
      <td>0.652830</td>
      <td>0.709821</td>
      <td>0.726415</td>
      <td>0.727520</td>
      <td>0.581197</td>
      <td>0.651042</td>
      <td>0.673835</td>
      <td>0.058635</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.10</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.1, 'gbs__n_estimators...</td>
      <td>0.732416</td>
      <td>0.705882</td>
      <td>0.660714</td>
      <td>0.777778</td>
      <td>0.521429</td>
      <td>0.576687</td>
      <td>...</td>
      <td>0.605</td>
      <td>0.673585</td>
      <td>0.727679</td>
      <td>0.683962</td>
      <td>0.673025</td>
      <td>0.602564</td>
      <td>0.671875</td>
      <td>0.669783</td>
      <td>0.060433</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.15</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.15, 'gbs__n_estimator...</td>
      <td>0.700306</td>
      <td>0.696594</td>
      <td>0.683673</td>
      <td>0.768889</td>
      <td>0.564286</td>
      <td>0.604294</td>
      <td>...</td>
      <td>0.600</td>
      <td>0.660377</td>
      <td>0.714286</td>
      <td>0.693396</td>
      <td>0.683924</td>
      <td>0.619658</td>
      <td>0.666667</td>
      <td>0.669566</td>
      <td>0.056325</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.05</td>
      <td>200</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.05, 'gbs__n_estimator...</td>
      <td>0.737003</td>
      <td>0.696594</td>
      <td>0.668367</td>
      <td>0.777778</td>
      <td>0.492857</td>
      <td>0.576687</td>
      <td>...</td>
      <td>0.600</td>
      <td>0.660377</td>
      <td>0.714286</td>
      <td>0.688679</td>
      <td>0.678474</td>
      <td>0.606838</td>
      <td>0.684896</td>
      <td>0.669083</td>
      <td>0.063933</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.05</td>
      <td>100</td>
      <td>88888888</td>
      <td>{'gbs__learning_rate': 0.05, 'gbs__n_estimator...</td>
      <td>0.723242</td>
      <td>0.662539</td>
      <td>0.665816</td>
      <td>0.760000</td>
      <td>0.500000</td>
      <td>0.536810</td>
      <td>...</td>
      <td>0.620</td>
      <td>0.635849</td>
      <td>0.705357</td>
      <td>0.681604</td>
      <td>0.675749</td>
      <td>0.574786</td>
      <td>0.653646</td>
      <td>0.658122</td>
      <td>0.067976</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>9 rows × 32 columns</p>
</div>




```python
##################################
# Identifying the best model
##################################
gbs_best_model_train_cv = gbs_grid_search.best_estimator_
print('Best Gradient Boosted Survival Model using the Cross-Validated Train Data: ')
print(f"Best Model Parameters: {gbs_grid_search.best_params_}")
```

    Best Gradient Boosted Survival Model using the Cross-Validated Train Data: 
    Best Model Parameters: {'gbs__learning_rate': 0.1, 'gbs__n_estimators': 200, 'gbs__random_state': 88888888}
    


```python
##################################
# Obtaining the cross-validation model performance of the 
# optimal Gradient Boosted Survival Model
# on the train set
##################################
optimal_gbs_heart_failure_y_crossvalidation_ci = gbs_grid_search.best_score_
print(f"Cross-Validation Concordance Index: {optimal_gbs_heart_failure_y_crossvalidation_ci}")
```

    Cross-Validation Concordance Index: 0.6765369976540313
    


```python
##################################
# Formulating a Gradient Boosted Survival Model
# with optimal hyperparameters
##################################
optimal_gbs_model = gbs_grid_search.best_estimator_
optimal_gbs_model.fit(X_train, y_train_array)
```




<style>#sk-container-id-10 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-10 {
  color: var(--sklearn-color-text);
}

#sk-container-id-10 pre {
  padding: 0;
}

#sk-container-id-10 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-10 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-10 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-10 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-10 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-10 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-10 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-10 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-10 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-10 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-10 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-10 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-10 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-10 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-10 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-10 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-10 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-10 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-10 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-10 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-10 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-10 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-10 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-10 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-10 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-10 div.sk-label label.sk-toggleable__label,
#sk-container-id-10 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-10 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-10 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-10 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-10 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-10 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-10 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-10 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-10 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-10 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-10 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-10 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-10 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-10" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;gbs&#x27;,
                 GradientBoostingSurvivalAnalysis(n_estimators=200,
                                                  random_state=88888888))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-69" type="checkbox" ><label for="sk-estimator-id-69" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;yeo_johnson&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numeric_predictors&#x27;,
                                                  PowerTransformer(),
                                                  [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                                   &#x27;SERUM_CREATININE&#x27;,
                                                   &#x27;SERUM_SODIUM&#x27;])])),
                (&#x27;gbs&#x27;,
                 GradientBoostingSurvivalAnalysis(n_estimators=200,
                                                  random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-70" type="checkbox" ><label for="sk-estimator-id-70" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;yeo_johnson: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for yeo_johnson: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;numeric_predictors&#x27;, PowerTransformer(),
                                 [&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;,
                                  &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-71" type="checkbox" ><label for="sk-estimator-id-71" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric_predictors</label><div class="sk-toggleable__content fitted"><pre>[&#x27;AGE&#x27;, &#x27;EJECTION_FRACTION&#x27;, &#x27;SERUM_CREATININE&#x27;, &#x27;SERUM_SODIUM&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-72" type="checkbox" ><label for="sk-estimator-id-72" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PowerTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>PowerTransformer()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-73" type="checkbox" ><label for="sk-estimator-id-73" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ANAEMIA&#x27;, &#x27;HIGH_BLOOD_PRESSURE&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-74" type="checkbox" ><label for="sk-estimator-id-74" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-75" type="checkbox" ><label for="sk-estimator-id-75" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">GradientBoostingSurvivalAnalysis</label><div class="sk-toggleable__content fitted"><pre>GradientBoostingSurvivalAnalysis(n_estimators=200, random_state=88888888)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Measuring model performance of the 
# optimal Gradient Boosted Survival Model
# on the train set
##################################
optimal_gbs_heart_failure_y_train_pred = optimal_gbs_model.predict(X_train)
optimal_gbs_heart_failure_y_train_ci = concordance_index_censored(y_train_array['DEATH_EVENT'], 
                                                                    y_train_array['TIME'], 
                                                                    optimal_gbs_heart_failure_y_train_pred)[0]
print(f"Apparent Concordance Index: {optimal_gbs_heart_failure_y_train_ci}")
```

    Apparent Concordance Index: 0.9274656878391319
    


```python
##################################
# Measuring model performance of the 
# optimal Gradient Boosted Survival Model
# on the validation set
##################################
optimal_gbs_heart_failure_y_validation_pred = optimal_gbs_model.predict(X_validation)
optimal_gbs_heart_failure_y_validation_ci = concordance_index_censored(y_validation_array['DEATH_EVENT'], 
                                                                         y_validation_array['TIME'], 
                                                                         optimal_gbs_heart_failure_y_validation_pred)[0]
print(f"Validation Concordance Index: {optimal_gbs_heart_failure_y_validation_ci}")
```

    Validation Concordance Index: 0.6575716234652115
    


```python
##################################
# Gathering the concordance indices
# from the train and tests sets for 
# Gradient Boosted Survival Model
##################################
gbs_set = pd.DataFrame(["Train","Cross-Validation","Validation"])
gbs_ci_values = pd.DataFrame([optimal_gbs_heart_failure_y_train_ci,
                              optimal_gbs_heart_failure_y_crossvalidation_ci,
                              optimal_gbs_heart_failure_y_validation_ci])
gbs_method = pd.DataFrame(["GBS"]*3)
gbs_summary = pd.concat([gbs_set, 
                           gbs_ci_values,
                           gbs_method], axis=1)
gbs_summary.columns = ['Set', 'Concordance.Index', 'Method']
gbs_summary.reset_index(inplace=True, drop=True)
display(gbs_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.927466</td>
      <td>GBS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.676537</td>
      <td>GBS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>0.657572</td>
      <td>GBS</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
heart_failure_validation.reset_index(drop=True, inplace=True)
kmf = KaplanMeierFitter()
heart_failure_validation['Predicted_Risks_GBS'] = optimal_gbs_heart_failure_y_validation_pred
heart_failure_validation['Predicted_RiskGroups_GBS'] = risk_groups = pd.qcut(heart_failure_validation['Predicted_Risks_GBS'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = heart_failure_validation[risk_groups == group]
    kmf.fit(group_data['TIME'], event_observed=group_data['DEATH_EVENT'], label=group)
    kmf.plot_survival_function()

plt.title('GBS Survival Probabilities by Predicted Risk Groups on Validation Set')
plt.xlabel('TIME')
plt.ylabel('DEATH_EVENT Survival Probability')
plt.show()
```


    
![png](output_226_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
validation_case_details = X_validation.iloc[[5, 10, 15, 20, 25]]
display(validation_case_details)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>60.0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>1.4</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>42.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0</td>
      <td>1.3</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50.0</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>57.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>1.1</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45.0</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0.8</td>
      <td>127.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Gathering the event and duration information
# for 5 test case samples
##################################
print(y_validation_array[[5, 10, 15, 20, 25]])
```

    [(False, 258) ( True,  65) (False,  90) (False,  79) ( True,  14)]
    


```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(heart_failure_validation.loc[[5, 10, 15, 20, 25]][['Predicted_RiskGroups_GBS']])
```

       Predicted_RiskGroups_GBS
    5                  Low-Risk
    10                High-Risk
    15                High-Risk
    20                High-Risk
    25                High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 validation cases
##################################
validation_case = X_validation.iloc[[5, 10, 15, 20, 25]]
validation_case_labels = ['Patient_5','Patient_10','Patient_15','Patient_20','Patient_25',]
validation_case_cumulative_hazard_function = optimal_gbs_model.predict_cumulative_hazard_function(validation_case)
validation_case_survival_function = optimal_gbs_model.predict_survival_function(validation_case)

fig, ax = plt.subplots(1,2,figsize=(17, 8))
for hazard_prediction, survival_prediction in zip(validation_case_cumulative_hazard_function, validation_case_survival_function):
    ax[0].step(hazard_prediction.x,hazard_prediction(hazard_prediction.x),where='post')
    ax[1].step(survival_prediction.x,survival_prediction(survival_prediction.x),where='post')
ax[0].set_title('GBS Cumulative Hazard for 5 Validation Cases')
ax[0].set_xlabel('TIME')
ax[0].set_ylim(0,2)
ax[0].set_ylabel('Cumulative Hazard')
ax[0].legend(validation_case_labels, loc="upper left")
ax[1].set_title('GBS Survival Function for 5 Validation Cases')
ax[1].set_xlabel('TIME')
ax[1].set_ylabel('DEATH_EVENT Survival Probability')
ax[1].legend(validation_case_labels, loc="lower left")
plt.show()
```


    
![png](output_230_0.png)
    



```python
##################################
# Saving the best Gradient Boosted Survival Model
# developed from the original training data
################################## 
joblib.dump(gbs_best_model_train_cv, 
            os.path.join("..", MODELS_PATH, "gbs_best_model.pkl"))
```




    ['..\\models\\gbs_best_model.pkl']



### 1.6.9 Model Selection <a class="anchor" id="1.6.9"></a>

1. The [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) was selected as the final model by demonstrating the best **concordance index** in the **validation data** with minimal overfitting between the apparent and cross-validated **train data**:
    * **train data (apparent)** = 0.7394
    * **train data (cross-validated)** = 0.7073
    * **validation data** = 0.7419
2. The optimal hyperparameters for the final model configuration was determined as follows:
    * <span style="color: #FF0000">alpha</span> = 10.00
3. The [cox net survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) also demonstrated comparably good survival prediction, but was not selected over the [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) due to model complexity.
4. The [survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html), [random survival forest model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html), and [gradient boosted survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) all showed conditions of overfitting as demonstrated by a considerable difference between the apparent and cross-validated **concordance index** values.



```python
##################################
# Gathering the concordance indices from 
# training, cross-validation and validation
##################################
set_labels = ['Train','Cross-Validation','Validation']
ci_plot = pd.DataFrame({'COXPH': list([optimal_coxph_heart_failure_y_train_ci,
                                       optimal_coxph_heart_failure_y_crossvalidation_ci,
                                       optimal_coxph_heart_failure_y_validation_ci]),
                        'COXNS': list([optimal_coxns_heart_failure_y_train_ci,
                                       optimal_coxns_heart_failure_y_crossvalidation_ci,
                                       optimal_coxns_heart_failure_y_validation_ci]),
                        'STREE': list([optimal_stree_heart_failure_y_train_ci,
                                       optimal_stree_heart_failure_y_crossvalidation_ci,
                                       optimal_stree_heart_failure_y_validation_ci]),
                        'RSF': list([optimal_rsf_heart_failure_y_train_ci,
                                     optimal_rsf_heart_failure_y_crossvalidation_ci,
                                     optimal_rsf_heart_failure_y_validation_ci]),
                        'GBS': list([optimal_gbs_heart_failure_y_train_ci,
                                     optimal_gbs_heart_failure_y_crossvalidation_ci,
                                     optimal_gbs_heart_failure_y_validation_ci])}, index = set_labels)
display(ci_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COXPH</th>
      <th>COXNS</th>
      <th>STREE</th>
      <th>RSF</th>
      <th>GBS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.741941</td>
      <td>0.741941</td>
      <td>0.799234</td>
      <td>0.871369</td>
      <td>0.927466</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.707318</td>
      <td>0.701369</td>
      <td>0.654169</td>
      <td>0.709097</td>
      <td>0.676537</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.739427</td>
      <td>0.729877</td>
      <td>0.644611</td>
      <td>0.693042</td>
      <td>0.657572</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the concordance indices
# for all models
##################################
ci_plot = ci_plot.plot.barh(figsize=(10, 6), width=0.90)
ci_plot.set_xlim(0.00,1.00)
ci_plot.set_title("Survival Prediction Model Comparison by Concordance Index")
ci_plot.set_xlabel("Concordance Index")
ci_plot.set_ylabel("Data Set")
ci_plot.grid(False)
ci_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in ci_plot.containers:
    ci_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_234_0.png)
    


### 1.6.10 Model Testing <a class="anchor" id="1.6.10"></a>

1. The selected [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) demonstrated sufficient **concordance index** in the independent **test data** :
    * **train data (apparent)** = 0.7394
    * **train data (cross-validated)** = 0.7073
    * **validation data** = 0.7419
    * **test data** = 0.7064
2. For benchmarking purposes, all candidate models were evaluated on the **test data**. Interestingly, the [survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html), [random survival forest model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html), and [gradient boosted survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) performed better than the selected model. In this case, the inconsistent performance (poor on validation, good on test) might be an indicator of instability. The [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) model (and to some extent, the [cox net survival model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) model), which shows more consistent performance across validation and test sets, is more reliable. Although, the selected model may not perform as well on the test set alone, its generalization across both validation and test sets makes it a more robust and stable choice in practice.



```python
##################################
# Evaluating the concordance indices
# on the test data
##################################
optimal_coxph_heart_failure_y_test_ci = concordance_index_censored(y_test_array['DEATH_EVENT'], 
                                                                   y_test_array['TIME'], 
                                                                   optimal_coxph_model.predict(X_test))[0]
optimal_coxns_heart_failure_y_test_ci = concordance_index_censored(y_test_array['DEATH_EVENT'], 
                                                                   y_test_array['TIME'], 
                                                                   optimal_coxns_model.predict(X_test))[0]
optimal_stree_heart_failure_y_test_ci = concordance_index_censored(y_test_array['DEATH_EVENT'], 
                                                                   y_test_array['TIME'], 
                                                                   optimal_stree_model.predict(X_test))[0]
optimal_rsf_heart_failure_y_test_ci = concordance_index_censored(y_test_array['DEATH_EVENT'], 
                                                                 y_test_array['TIME'], 
                                                                 optimal_rsf_model.predict(X_test))[0]
optimal_gbs_heart_failure_y_test_ci = concordance_index_censored(y_test_array['DEATH_EVENT'], 
                                                                 y_test_array['TIME'], 
                                                                 optimal_gbs_model.predict(X_test))[0]
```


```python
##################################
# Adding the the concordance index estimated
# from the test data
##################################
set_labels = ['Train','Cross-Validation','Validation','Test']
updated_ci_plot = pd.DataFrame({'COXPH': list([optimal_coxph_heart_failure_y_train_ci,
                                               optimal_coxph_heart_failure_y_crossvalidation_ci,
                                               optimal_coxph_heart_failure_y_validation_ci,
                                               optimal_coxph_heart_failure_y_test_ci]),
                                'COXNS': list([optimal_coxns_heart_failure_y_train_ci,
                                               optimal_coxns_heart_failure_y_crossvalidation_ci,
                                               optimal_coxns_heart_failure_y_validation_ci,
                                               optimal_coxns_heart_failure_y_test_ci]),
                                'STREE': list([optimal_stree_heart_failure_y_train_ci,
                                               optimal_stree_heart_failure_y_crossvalidation_ci,
                                               optimal_stree_heart_failure_y_validation_ci,
                                               optimal_stree_heart_failure_y_test_ci]),
                                'RSF': list([optimal_rsf_heart_failure_y_train_ci,
                                             optimal_rsf_heart_failure_y_crossvalidation_ci,
                                             optimal_rsf_heart_failure_y_validation_ci,
                                             optimal_rsf_heart_failure_y_test_ci]),
                                'GBS': list([optimal_gbs_heart_failure_y_train_ci,
                                             optimal_gbs_heart_failure_y_crossvalidation_ci,
                                             optimal_gbs_heart_failure_y_validation_ci,
                                             optimal_gbs_heart_failure_y_test_ci])}, index = set_labels)
display(updated_ci_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COXPH</th>
      <th>COXNS</th>
      <th>STREE</th>
      <th>RSF</th>
      <th>GBS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.741941</td>
      <td>0.741941</td>
      <td>0.799234</td>
      <td>0.871369</td>
      <td>0.927466</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.707318</td>
      <td>0.701369</td>
      <td>0.654169</td>
      <td>0.709097</td>
      <td>0.676537</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.739427</td>
      <td>0.729877</td>
      <td>0.644611</td>
      <td>0.693042</td>
      <td>0.657572</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.706422</td>
      <td>0.719831</td>
      <td>0.762526</td>
      <td>0.760056</td>
      <td>0.778758</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the concordance indices
# for all models
##################################
updated_ci_plot = updated_ci_plot.plot.barh(figsize=(10, 8), width=0.90)
updated_ci_plot.set_xlim(0.00,1.00)
updated_ci_plot.set_title("Survival Prediction Model Comparison by Concordance Index")
updated_ci_plot.set_xlabel("Concordance Index")
updated_ci_plot.set_ylabel("Data Set")
updated_ci_plot.grid(False)
updated_ci_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in updated_ci_plot.containers:
    updated_ci_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_238_0.png)
    


### 1.6.11 Model Inference <a class="anchor" id="1.6.11"></a>

1. For the final selected survival prediction model developed from the **train data**, the contributions of the predictors, ranked by importance, are given as follows:
    * [Cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) 
        * <span style="color: #FF0000">SERUM_CREATININE</span>
        * <span style="color: #FF0000">EJECTION_FRACTION</span>
        * <span style="color: #FF0000">SERUM_SODIUM</span>
        * <span style="color: #FF0000">ANAEMIA</span>
        * <span style="color: #FF0000">AGE</span>
        * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
2. Model inference involved indicating the characteristics and predicting the survival probability of the new case against the model training observations.
    * Characteristics based on all predictors used for generating the final selected survival prediction model
    * Predicted heart failure survival probability profile based on the final selected survival prediction model


```python
##################################
# Determining the Cox Proportional Hazards Regression model
# absolute coefficient-based feature importance 
# on train data
##################################
coxph_train_feature_importance = pd.DataFrame(
    {'Signed.Coefficient': optimal_coxph_model.named_steps['coxph'].coef_,
    'Absolute.Coefficient': np.abs(optimal_coxph_model.named_steps['coxph'].coef_)}, index=X_train.columns)
display(coxph_train_feature_importance.sort_values('Absolute.Coefficient', ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Signed.Coefficient</th>
      <th>Absolute.Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EJECTION_FRACTION</th>
      <td>0.407833</td>
      <td>0.407833</td>
    </tr>
    <tr>
      <th>SERUM_CREATININE</th>
      <td>0.352092</td>
      <td>0.352092</td>
    </tr>
    <tr>
      <th>ANAEMIA</th>
      <td>-0.306170</td>
      <td>0.306170</td>
    </tr>
    <tr>
      <th>HIGH_BLOOD_PRESSURE</th>
      <td>-0.280524</td>
      <td>0.280524</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.245804</td>
      <td>0.245804</td>
    </tr>
    <tr>
      <th>SERUM_SODIUM</th>
      <td>0.234638</td>
      <td>0.234638</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the Cox Proportional Hazards Regression model
# absolute coefficient-based feature importance 
# on train data
##################################
coxph_train_coefficient_importance_summary = coxph_train_feature_importance.sort_values('Absolute.Coefficient', ascending=True)
plt.figure(figsize=(17, 8))
plt.barh(coxph_train_coefficient_importance_summary.index, coxph_train_coefficient_importance_summary['Absolute.Coefficient'])
plt.xlabel('Predictor Contribution: Absolute Coefficient')
plt.ylabel('Predictor')
plt.title('Feature Importance - Final Survival Prediction Model: Cox Proportional Hazards Regression')
plt.tight_layout()
plt.show()
```


    
![png](output_241_0.png)
    



```python
##################################
# Determining the Cox Proportional Hazards Regression model
# permutation-based feature importance 
# on train data
##################################
coxph_train_feature_importance = permutation_importance(optimal_coxph_model,
                                                        X_train, 
                                                        y_train_array, 
                                                        n_repeats=15, 
                                                        random_state=88888888)

coxph_train_feature_importance_summary = pd.DataFrame(
    {k: coxph_train_feature_importance[k]
     for k in ("importances_mean", "importances_std")}, 
    index=X_train.columns).sort_values(by="importances_mean", ascending=False)
coxph_train_feature_importance_summary.columns = ['Importances.Mean', 'Importances.Std']
display(coxph_train_feature_importance_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importances.Mean</th>
      <th>Importances.Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SERUM_CREATININE</th>
      <td>0.055362</td>
      <td>0.017600</td>
    </tr>
    <tr>
      <th>EJECTION_FRACTION</th>
      <td>0.032078</td>
      <td>0.014803</td>
    </tr>
    <tr>
      <th>SERUM_SODIUM</th>
      <td>0.023034</td>
      <td>0.012928</td>
    </tr>
    <tr>
      <th>ANAEMIA</th>
      <td>0.018449</td>
      <td>0.009402</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.017763</td>
      <td>0.009435</td>
    </tr>
    <tr>
      <th>HIGH_BLOOD_PRESSURE</th>
      <td>0.002915</td>
      <td>0.004219</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the Cox Proportional Hazards Regression model
# absolute coefficient-based feature importance 
# on train data
##################################
coxph_train_feature_importance_summary = coxph_train_feature_importance_summary.sort_values('Importances.Mean', ascending=True)
plt.figure(figsize=(17, 8))
plt.barh(coxph_train_feature_importance_summary.index, coxph_train_feature_importance_summary['Importances.Mean'])
plt.xlabel('Predictor Contribution: Permutation Importance')
plt.ylabel('Predictor')
plt.title('Feature Importance - Final Survival Prediction Model: Cox Proportional Hazards Regression')
plt.tight_layout()
plt.show()
```


    
![png](output_243_0.png)
    



```python
##################################
# Rebuilding the training data
# for plotting kaplan-meier charts
##################################
X_train_indices = X_train.index.tolist()
heart_failure_MI = heart_failure_EDA.copy()
heart_failure_MI = heart_failure_MI.drop(['DIABETES','SEX', 'SMOKING', 'CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
heart_failure_MI = heart_failure_MI.loc[X_train_indices]
heart_failure_MI.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>DEATH_EVENT</th>
      <th>TIME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>266</th>
      <td>-0.423454</td>
      <td>-1.773346</td>
      <td>1.144260</td>
      <td>-0.689301</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>True</td>
      <td>241.0</td>
    </tr>
    <tr>
      <th>180</th>
      <td>-2.043070</td>
      <td>-0.633046</td>
      <td>-0.732811</td>
      <td>-0.244181</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>False</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0.434332</td>
      <td>-0.160461</td>
      <td>-0.087641</td>
      <td>1.348555</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>False</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>-1.446547</td>
      <td>-1.163741</td>
      <td>-1.149080</td>
      <td>-0.471658</td>
      <td>Present</td>
      <td>Absent</td>
      <td>False</td>
      <td>230.0</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1.173233</td>
      <td>1.021735</td>
      <td>-0.087641</td>
      <td>3.397822</td>
      <td>Absent</td>
      <td>Present</td>
      <td>False</td>
      <td>209.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Determining the medians for the numeric predictors
##################################
heart_failure_MI_numeric = heart_failure_MI[["AGE","EJECTION_FRACTION","SERUM_CREATININE","SERUM_SODIUM"]]
numeric_predictor_median_list = heart_failure_MI_numeric.median()
numeric_predictor_median_list
```




    AGE                  0.065124
    EJECTION_FRACTION    0.100914
    SERUM_CREATININE    -0.087641
    SERUM_SODIUM        -0.006503
    dtype: float64




```python
##################################
# Creating a function to bin
# numeric predictors into two groups
##################################
def bin_numeric_model_predictor(df, predictor):
    median = numeric_predictor_median_list.loc[predictor]
    df[predictor] = np.where(df[predictor] <= median, "Low", "High")
    return df
```


```python
##################################
# Binning the numeric predictors
# into two groups
##################################
for numeric_column in ["AGE","EJECTION_FRACTION","SERUM_CREATININE","SERUM_SODIUM"]:
    heart_failure_MI_EDA = bin_numeric_model_predictor(heart_failure_MI, numeric_column)
```


```python
##################################
# Exploring the transformed
# dataset for plotting
##################################
heart_failure_MI_EDA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>DEATH_EVENT</th>
      <th>TIME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>266</th>
      <td>Low</td>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>True</td>
      <td>241.0</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Low</td>
      <td>Low</td>
      <td>Low</td>
      <td>Low</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>False</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>288</th>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>High</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>False</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>Low</td>
      <td>Low</td>
      <td>Low</td>
      <td>Low</td>
      <td>Present</td>
      <td>Absent</td>
      <td>False</td>
      <td>230.0</td>
    </tr>
    <tr>
      <th>236</th>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
      <td>High</td>
      <td>Absent</td>
      <td>Present</td>
      <td>False</td>
      <td>209.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining a function to plot the
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

    # Plotting each category with a partly red or blue transparent line
    for value in categories:
        mask = df[cat_var] == value
        kmf.fit(df['TIME'][mask], event_observed=df['DEATH_EVENT'][mask], label=f'{cat_var}={value} (Baseline Distribution)')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[str(value)], linestyle='-', linewidth=6.0, alpha=0.30)

    # Overlaying a black broken line for the new case if provided
    if new_case_value is not None:
        mask_new_case = df[cat_var] == new_case_value
        kmf.fit(df['TIME'][mask_new_case], event_observed=df['DEATH_EVENT'][mask_new_case], label=f'{cat_var}={new_case_value} (Test Case)')
        kmf.plot_survival_function(ax=ax, ci_show=False, color='black', linestyle=':', linewidth=3.0)
```


```python
##################################
# Plotting the estimated survival profiles
# of the model training data
# using Kaplan-Meier Plots
##################################
fig, axes = plt.subplots(3, 2, figsize=(17, 18))

heart_failure_predictors = ['AGE','EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM','ANAEMIA','HIGH_BLOOD_PRESSURE']

for i, predictor in enumerate(heart_failure_predictors):
    ax = axes[i // 2, i % 2]
    plot_kaplan_meier(heart_failure_MI_EDA, predictor, ax, new_case_value=None)
    ax.set_title(f'DEATH_EVENT Survival Probabilities by {predictor} Categories')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Estimated Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_250_0.png)
    



```python
##################################
# Estimating the survival functions
# for the training data
##################################
heart_failure_train_survival_function = optimal_coxph_model.predict_survival_function(X_train)
```


```python
##################################
# Resetting the index for 
# plotting survival functions
# for the training data
##################################
y_train_reset_index = y_train.reset_index()
```


```python
##################################
# Plotting the baseline survival functions
# for the training data
##################################
plt.figure(figsize=(17, 8))
for i, surv_func in enumerate(heart_failure_train_survival_function):
    plt.step(surv_func.x, 
             surv_func.y, 
             where="post", 
             color='red' if y_train_reset_index['DEATH_EVENT'][i] == 1 else 'blue', 
             linewidth=6.0,
             alpha=0.05)
red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
plt.legend(handles=[red_patch, blue_patch], facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
plt.title('Final Survival Prediction Model: Cox Proportional Hazards Regression')
plt.xlabel('Time (Days)')
plt.ylabel('Estimated Survival Probability')
plt.tight_layout(rect=[0, 0, 1.00, 0.95])
plt.show()
```


    
![png](output_253_0.png)
    



```python
##################################
# Describing the details of the 
# test case for evaluation
##################################
X_sample = {'AGE': 43,  
            'ANAEMIA': 0, 
            'EJECTION_FRACTION': 75,
            'HIGH_BLOOD_PRESSURE': 1,
            'SERUM_CREATININE': 0.75, 
            'SERUM_SODIUM': 100}
X_test_sample = pd.DataFrame([X_sample])
X_test_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>ANAEMIA</th>
      <th>EJECTION_FRACTION</th>
      <th>HIGH_BLOOD_PRESSURE</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43</td>
      <td>0</td>
      <td>75</td>
      <td>1</td>
      <td>0.75</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying preprocessing to the new case
##################################
coxph_pipeline.fit(X_train, y_train_array)
X_test_sample_transformed = coxph_pipeline.named_steps['yeo_johnson'].transform(X_test_sample)
X_test_sample_converted = pd.DataFrame([X_test_sample_transformed[0]], columns=["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM", "ANAEMIA", "HIGH_BLOOD_PRESSURE"])
X_test_sample_converted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>HIGH_BLOOD_PRESSURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.669035</td>
      <td>2.423321</td>
      <td>-1.404833</td>
      <td>-4.476082</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Binning numeric predictors into two groups
##################################
for i, col in enumerate(["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM"]):
    X_test_sample_converted[col] = X_test_sample_converted[col].apply(lambda x: 'High' if x > numeric_predictor_median_list[i] else 'Low')
```


```python
##################################
# Converting integer predictors into labels
##################################
for col in ["ANAEMIA", "HIGH_BLOOD_PRESSURE"]:
    X_test_sample_converted[col] = X_test_sample_converted[col].apply(lambda x: 'Absent' if x < 1.0 else 'Present')
```


```python
##################################
# Describing the details of the 
# test case for evaluation
##################################
X_test_sample_converted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
      <th>ANAEMIA</th>
      <th>HIGH_BLOOD_PRESSURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>High</td>
      <td>Low</td>
      <td>Low</td>
      <td>Absent</td>
      <td>Present</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the estimated survival profiles
# of the test case
# using Kaplan-Meier Plots
##################################
fig, axes = plt.subplots(3, 2, figsize=(17, 18))

heart_failure_predictors = ['AGE','EJECTION_FRACTION','SERUM_CREATININE','SERUM_SODIUM','ANAEMIA','HIGH_BLOOD_PRESSURE']

for i, predictor in enumerate(heart_failure_predictors):
    ax = axes[i // 2, i % 2]
    plot_kaplan_meier(heart_failure_MI_EDA, predictor, ax, new_case_value=X_test_sample_converted[predictor][0])
    ax.set_title(f'DEATH_EVENT Survival Probabilities by {predictor} Categories')
    ax.set_xlabel('TIME')
    ax.set_ylabel('DEATH_EVENT Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_259_0.png)
    



```python
##################################
# Computing the estimated survival probability
# for the test case
##################################
X_test_sample_survival_function = optimal_coxph_model.predict_survival_function(X_test_sample)
```


```python
##################################
# Plotting the estimated survival probability
# for the test case 
# in the baseline survival function
# of the final survival prediction model
##################################
plt.figure(figsize=(17, 8))
for i, surv_func in enumerate(heart_failure_train_survival_function):
    plt.step(surv_func.x, 
             surv_func.y, 
             where="post", 
             color='red' if y_train_reset_index['DEATH_EVENT'][i] == 1 else 'blue', 
             linewidth=6.0,
             alpha=0.05)
plt.step(X_test_sample_survival_function[0].x, 
         X_test_sample_survival_function[0].y, 
         where="post", 
         color='black', 
         linewidth=3.0, 
         linestyle=':',
         label='Test Case')
red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
black_patch = plt.Line2D([0], [0], color='black', lw=3, linestyle=":", label='Test Case')
plt.legend(handles=[red_patch, blue_patch, black_patch], facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
plt.title('Final Survival Prediction Model: Cox Proportional Hazards Regression')
plt.xlabel('Time (Days)')
plt.ylabel('Estimated Survival Probability')
plt.tight_layout(rect=[0, 0, 1.00, 0.95])
plt.show()
```


    
![png](output_261_0.png)
    



```python
##################################
# Determining the risk category
# for the test case
##################################
optimal_coxph_heart_failure_y_train_pred = optimal_coxph_model.predict(X_train)
heart_failure_train['Predicted_Risks_CoxPH'] = optimal_coxph_heart_failure_y_train_pred
risk_groups, risk_group_bin_range = pd.qcut(heart_failure_train['Predicted_Risks_CoxPH'], 2, labels=['Low-Risk', 'High-Risk'], retbins=True)
risk_group_threshold = risk_group_bin_range[1]
X_test_sample_risk_category = "High-Risk" if (optimal_coxph_model.predict(X_test_sample) > risk_group_threshold) else "Low-Risk"
```


```python
##################################
# Computing the estimated survival probabilities
# for the test case at five defined time points
##################################
X_test_sample_survival_time = np.array([50, 100, 150, 200, 250])
X_test_sample_survival_probability = np.interp(X_test_sample_survival_time, 
                                               X_test_sample_survival_function[0].x, 
                                               X_test_sample_survival_function[0].y)
X_test_sample_survival_probability = X_test_sample_survival_probability*100
for survival_time, survival_probability in zip(X_test_sample_survival_time, X_test_sample_survival_probability):
    print(f"Test Case Survival Probability ({survival_time} Days): {survival_probability:.2f}%")
print(f"Test Case Risk Category: {X_test_sample_risk_category}")
```

    Test Case Survival Probability (50 Days): 90.98%
    Test Case Survival Probability (100 Days): 87.65%
    Test Case Survival Probability (150 Days): 84.60%
    Test Case Survival Probability (200 Days): 78.48%
    Test Case Survival Probability (250 Days): 70.70%
    Test Case Risk Category: Low-Risk
    

## 1.7. Predictive Model Deployment Using Streamlit and Streamlit Community Cloud <a class="anchor" id="1.7"></a>

### 1.7.1 Model Prediction Application Code Development <a class="anchor" id="1.7.1"></a>

### 1.7.2 Model Application Programming Interface Code Development <a class="anchor" id="1.7.2"></a>

### 1.7.3 User Interface Application Code Development <a class="anchor" id="1.7.3"></a>

### 1.7.4 Web Application <a class="anchor" id="1.7.4"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>


* **[Book]** [Clinical Prediction Models](http://clinicalpredictionmodels.org/) by Ewout Steyerberg
* **[Book]** [Survival Analysis: A Self-Learning Text](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) by David Kleinbaum and Mitchel Klein
* **[Book]** [Applied Survival Analysis Using R](https://link.springer.com/book/10.1007/978-3-319-31245-3/) by Dirk Moore
* **[Book]** [Survival Analysis with Python](https://www.taylorfrancis.com/books/mono/10.1201/9781003255499/survival-analysis-python-avishek-nag) by Avishek Nag
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Python Library API]** [SciKit-Learn](https://scikit-learn.org/stable/index.html) by SciKit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [Lifelines](https://lifelines.readthedocs.io/en/latest/) by Lifelines Team
* **[Kaggle Project]** [Applied Reliability, Solutions To Problems](https://www.kaggle.com/code/keenanzhuo/applied-reliability-solutions-to-problems) by Keenan Zhuo (Kaggle)
* **[Kaggle Project]** [Survival Models VS ML Models Benchmark - Churn Tel](https://www.kaggle.com/code/caralosal/survival-models-vs-ml-models-benchmark-churn-tel) by Carlos Alonso Salcedo (Kaggle)
* **[Kaggle Project]** [Survival Analysis with Cox Model Implementation](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Bryan Boulé (Kaggle)
* **[Kaggle Project]** [Survival Analysis](https://www.kaggle.com/code/gunesevitan/survival-analysis/notebook) by Gunes Evitan (Kaggle)
* **[Kaggle Project]** [Survival Analysis of Lung Cancer Patients](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Sayan Chakraborty (Kaggle)
* **[Kaggle Project]** [COVID-19 Cox Survival Regression](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Ilias Katsabalos (Kaggle)
* **[Kaggle Project]** [Liver Cirrhosis Prediction with XGboost & EDA](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) by Arjun Bhaybang (Kaggle)
* **[Article]** [Exploring Time-to-Event with Survival Analysis](https://towardsdatascience.com/exploring-time-to-event-with-survival-analysis-8b0a7a33a7be) by Olivia Tanuwidjaja (Towards Data Science)
* **[Article]** [The Complete Introduction to Survival Analysis in Python](https://towardsdatascience.com/the-complete-introduction-to-survival-analysis-in-python-7523e17737e6) by Marco Peixeiro (Towards Data Science)
* **[Article]** [Survival Analysis Simplified: Explaining and Applying with Python](https://medium.com/@zynp.atlii/survival-analysis-simplified-explaining-and-applying-with-python-7efacf86ba32) by Zeynep Atli (Towards Data Science)
* **[Article]** [Survival Analysis in Python (KM Estimate, Cox-PH and AFT Model)](https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d) by Rahul Raoniar (Medium)
* **[Article]** [How to Evaluate Survival Analysis Models)](https://towardsdatascience.com/how-to-evaluate-survival-analysis-models-dd67bc10caae) by Nicolo Cosimo Albanese (Towards Data Science)
* **[Article]** [Survival Analysis with Python Tutorial — How, What, When, and Why)](https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312) by Towards AI Team (Medium)
* **[Article]** [Survival Analysis: Predict Time-To-Event With Machine Learning)](https://towardsdatascience.com/survival-analysis-predict-time-to-event-with-machine-learning-part-i-ba52f9ab9a46) by Lina Faik (Medium)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 1](https://www.kdnuggets.com/2020/07/complete-guide-survival-analysis-python-part1.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 2](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-2.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 3](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Pratik Shukla (KDNuggets)
* **[Article]** [Model Explainability using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)](https://medium.com/@anshulgoel991/model-exploitability-using-shap-shapley-additive-explanations-and-lime-local-interpretable-cb4f5594fc1a) by Anshul Goel (Medium)
* **[Article]** [A Comprehensive Guide into SHAP (SHapley Additive exPlanations) Values](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Brain John Aboze (DeepChecks.Com)
* **[Article]** [SHAP - Understanding How This Method for Explainable AI Works](https://safjan.com/how-the-shap-method-for-explainable-ai-works/#google_vignette) by Krystian Safjan (Safjan.Com)
* **[Article]** [SHAP: Shapley Additive Explanations](https://towardsdatascience.com/shap-shapley-additive-explanations-5a2a271ed9c3) by Fernando Lopez (Medium)
* **[Article]** [Explainable Machine Learning, Game Theory, and Shapley Values: A technical review](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Soufiane Fadel (Statistics Canada)
* **[Article]** [SHAP Values Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30) by Samuele Mazzanti (Towards Data Science)
* **[Article]** [Explaining Machine Learning Models: A Non-Technical Guide to Interpreting SHAP Analyses](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/) by Aidan Cooper (AidanCooper.Co.UK)
* **[Article]** [Shapley Additive Explanations: Unveiling the Black Box of Machine Learning](https://python.plainenglish.io/shapley-additive-explanations-unveiling-the-black-box-of-machine-learning-477ba01ffa07) by Evertone Gomede (Medium)
* **[Article]** [SHAP (SHapley Additive exPlanations)](https://www.nerd-data.com/shap/) by Narut Soontranon (Nerd-Data.Com)
* **[Article]** [Survival Analysis](https://quantdev.ssri.psu.edu/resources/survival-analysis) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 1: How to Format Data for Several Types of Survival Analysis Models](https://quantdev.ssri.psu.edu/tutorials/part-1-how-format-data-several-types-survival-analysis-models) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 2: Single-Episode Cox Regression Model with Time-Invariant Predictors](https://quantdev.ssri.psu.edu/tutorials/part-2-single-episode-cox-regression-model-time-invariant-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 3: Single-Episode Cox Regression Model with Time-Varying Predictors](https://quantdev.ssri.psu.edu/tutorials/part-3-single-episode-cox-regression-model-time-varying-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 4: Recurring-Episode Cox Regression Model with Time-Invariant Predictors](https://quantdev.ssri.psu.edu/tutorials/part-4-recurring-episode-cox-regression-model-time-invariant-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 5: Recurring-Episode Cox Regression Model with Time-Varying Predictors](https://quantdev.ssri.psu.edu/tutorials/part-5-recurring-episode-cox-regression-model-time-varying-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Parametric Survival Modeling](https://devinincerti.com/2019/06/18/parametric_survival.html) by Devin Incerti (DevinIncerti.Com)
* **[Article]** [Survival Analysis Simplified: Explaining and Applying with Python](https://medium.com/@zynp.atlii/survival-analysis-simplified-explaining-and-applying-with-python-7efacf86ba32) by Zeynep Atli (Medium)
* **[Article]** [Understanding Survival Analysis Models: Bridging the Gap between Parametric and Semiparametric Approaches](https://medium.com/@zynp.atlii/understanding-survival-analysis-models-bridging-the-gap-between-parametric-and-semiparametric-923cdcfc9f05) by Zeynep Atli (Medium)
* **[Article]** [Survival Modeling — Accelerated Failure Time — XGBoost](https://towardsdatascience.com/survival-modeling-accelerated-failure-time-xgboost-971aaa1ba794) by Avinash Barnwal (Medium)
* **[Publication]** [Regression Models and Life Tables](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) by David Cox (Royal Statistical Society)
* **[Publication]** [Covariance Analysis of Censored Survival Data](https://pubmed.ncbi.nlm.nih.gov/4813387/) by Norman Breslow (Biometrics)
* **[Publication]** [The Efficiency of Cox’s Likelihood Function for Censored Data](https://www.jstor.org/stable/2286217) by Bradley Efron (Journal of the American Statistical Association)
* **[Publication]** [Regularization Paths for Cox’s Proportional Hazards Model via Coordinate Descent](https://doi.org/10.18637/jss.v039.i05) by Noah Simon, Jerome Friedman, Trevor Hastie and Rob Tibshirani (Journal of Statistical Software)
* **[Publication]** [Shapley Additive Explanations](https://dl.acm.org/doi/10.5555/1756006.1756007) by Noah Simon, Jerome Friedman, Trevor Hastie and Rob Tibshirani (Journal of Statistical Software) by Erik Strumbelj and Igor Kononenko (The Journal of Machine Learning Research)
* **[Publication]** [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) by Scott Lundberg and Sun-In Lee (Conference on Neural Information Processing Systems)
* **[Publication]** [Survival Analysis Part I: Basic Concepts and First Analyses](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394262/) by Taane Clark (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part II: Multivariate Data Analysis – An Introduction to Concepts and Methods](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part III: Multivariate Data Analysis – Choosing a Model and Assessing its Adequacy and Fit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2376927/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part IV: Further Concepts and Methods in Survival Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394469/) by Taane Clark (British Journal of Cancer)
* **[Publication]** [Marginal Likelihoods Based on Cox's Regression and Life Model](https://www.jstor.org/stable/2334538) by Jack Kalbfleisch and Ross Prentice (Biometrika)
* **[Publication]** [Hazard Rate Models with Covariates](https://www.jstor.org/stable/2529934) by Jack Kalbfleisch and Ross Prentice (Biometrics)
* **[Publication]** [Linear Regression with Censored Data](https://www.jstor.org/stable/2335161) by Jonathan Buckley and Ian James (Biometrika)
* **[Publication]** [A Statistical Distribution Function of Wide Applicability](https://www.semanticscholar.org/paper/A-Statistical-Distribution-Function-of-Wide-Weibull/88c37770028e7ed61180a34d6a837a9a4db3b264) by Waloddi Weibull (Journal of Applied Mechanics)
* **[Publication]** [Exponential Survivals with Censoring and Explanatory Variables](https://www.jstor.org/stable/2334539) by Ross Prentice (Biometrika)
* **[Publication]** [The Lognormal Distribution, with Special Reference to its Uses in Economics](https://www.semanticscholar.org/paper/The-Lognormal-Distribution%2C-with-Special-Reference-Corlett-Aitchison/1f59c53ff512fa77e7aee5e6d98b1786c2aaf129) by John Aitchison and James Brown (Economics Applied Statistics)
* **[Course]** [Survival Analysis in Python](https://app.datacamp.com/learn/courses/survival-analysis-in-python) by Shae Wang (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

