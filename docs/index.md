***
# Model Deployment : Estimating Heart Failure Survival Risk Profiles From Cardiovascular, Hematologic And Metabolic Markers

***
### John Pauline Pineda <br> <br> *September 21, 2024*
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
        * [1.6.4 Semi-Parametric Model Fitting | Hyperparameter Tuning | Validation](#1.6.4)
        * [1.6.5 Parametric Model Fitting | Hyperparameter Tuning | Validation](#1.6.5)
        * [1.6.6 Model Selection](#1.6.7)
        * [1.6.7 Model Testing](#1.6.8)
        * [1.6.8 Model Inference](#1.6.9)
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
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import pointbiserialr

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
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
heart_failure_filtered = heart_failure_transformed.drop(['DIABETES','SEX', 'SMOKING', 'AGE','CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
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
    </tr>
    <tr>
      <th>294</th>
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
<p>299 rows × 7 columns</p>
</div>


### 1.6.2 Data Splitting <a class="anchor" id="1.6.2"></a>


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
    


    (299, 7)



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
X_train_initial = heart_failure_train_initial.drop('DEATH_EVENT', axis = 1)
y_train_initial = heart_failure_train_initial['DEATH_EVENT']
print('Initial Training Dataset Dimensions: ')
display(X_train_initial.shape)
display(y_train_initial.shape)
print('Initial Training Target Variable Breakdown: ')
display(y_train_initial.value_counts())
print('Initial Training Target Variable Proportion: ')
display(y_train_initial.value_counts(normalize = True))
```

    Initial Training Dataset Dimensions: 
    


    (224, 6)



    (224,)


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
X_test = heart_failure_test.drop('DEATH_EVENT', axis = 1)
y_test = heart_failure_test['DEATH_EVENT']
print('Test Dataset Dimensions: ')
display(X_test.shape)
display(y_test.shape)
print('Test Target Variable Breakdown: ')
display(y_test.value_counts())
print('Test Target Variable Proportion: ')
display(y_test.value_counts(normalize = True))
```

    Test Dataset Dimensions: 
    


    (75, 6)



    (75,)


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
X_train = heart_failure_train.drop('DEATH_EVENT', axis = 1)
y_train = heart_failure_train['DEATH_EVENT']
print('Final Training Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Final Training Target Variable Breakdown: ')
display(y_train.value_counts())
print('Final Training Target Variable Proportion: ')
display(y_train.value_counts(normalize = True))
```

    Final Training Dataset Dimensions: 
    


    (168, 6)



    (168,)


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
X_validation = heart_failure_validation.drop('DEATH_EVENT', axis = 1)
y_validation = heart_failure_validation['DEATH_EVENT']
print('Validation Dataset Dimensions: ')
display(X_validation.shape)
display(y_validation.shape)
print('Validation Target Variable Breakdown: ')
display(y_validation.value_counts())
print('Validation Target Variable Proportion: ')
display(y_validation.value_counts(normalize = True))
```

    Validation Dataset Dimensions: 
    


    (56, 6)



    (56,)


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

### 1.6.3 Modelling Pipeline Development <a class="anchor" id="1.6.3"></a>

### 1.6.4 Semi-Parametric Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.4"></a>

### 1.6.5 Model Fitting using Original Training Data | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.5"></a>

### 1.6.6 Model Selection <a class="anchor" id="1.6.6"></a>

### 1.6.7 Model Testing <a class="anchor" id="1.6.7"></a>

### 1.6.8 Model Inference <a class="anchor" id="1.6.8"></a>

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

