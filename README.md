# Scalable_Machine_Learning

# Installation

    conda env create --file environment.yml 
    conda activate ID2223_Project


# Project Description:

The dataset is the [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data). 

We generate new data in two ways:
1) We use a GAN to generate synthetic data. We use the [Ydata-synthetic](https://docs.synthetic.ydata.ai/1.3/) library to generate synthetic data. We run a daily feature pipeline to add the synthetic data to our feature store.  
2) In addition, we allow users to input their own data. We use the [Hopswork](https://www.hopsworks.ai/) platform to store the user data in a feature group for only user data. And we run a daily feature pipeline to add the user data to our feature store.  

Step 2 was suggested by the examiner:
"I suggest that users can enter their data (and the outcome - if they have heart disease, if they know). Then you can write a feature pipeline to collect that in a feature store. That way you will have a real non-static dataset, where new data comes from users."

We opted to keeping the Sythetic data as well because we believe that the data mimicks real data successfully and allows for interesting monitoring analysis.

# Prediction Problem:
The prediction problem is whether a given patient has a risk of heart disease or not. We use Sklearn as our Model

# Technologies Used:
[Scikit-Learn](https://scikit-learn.org/stable/index.html) - ML Models  
[Ydata-synthetic](https://docs.synthetic.ydata.ai/1.3/) - GAN  
[Hopswork](https://www.hopsworks.ai/) - Model Registry and Feature Store  
[Modal](https://modal.com/)  - CRON job for Synthetic Data  
[Hugging Face](https://huggingface.co/) - UI
[SHAP](https://shap.readthedocs.io/en/latest/) - Explainable AI  
[Imbalanced-Learn](https://imbalanced-learn.org/stable/) - Library to deal with Imbalanced Classes  

# Extra For Excellent 

## 1. Explainable AI

We use the [SHAP](https://shap.readthedocs.io/en/latest/) library to explain the predictions of our model. The full capabilities of SHAP are visible in [playing around](playing_around.ipynb) notebook. However, the library had certain limitations, which resulted in us only using a subset of the features in the final SERVERLESS project. Specifically, the library could not generate advanced plots explaining all predictions (we showcase this in playing around). Nontheless, it gives a lot of good information for the user to understand the model.

## 2. Imbalanced Classes

The dataset was severely imbalanced. We used the [Imbalanced-Learn](https://imbalanced-learn.org/stable/) library to deal with this. We used the SMOTE algorithm to oversample the minority class. If we did not perform oversampling, the model would predict all patients to be healthy, which would result in a high accuracy, but a useless model. Therefore, we oversampled the minority class to get a more balanced dataset.

## 3. Dealing with Missing Values
We allow users to input "Unknown" information for certain features. We deal with this by imputing the missing values with the most frequent value for that feature. This allows users to still receive a prediction of their results even if they do not know all the information.

# Documentation


## 1_heart_eda_and_feature_group

This notebook contains the EDA and Feature Group creation. It also contains the code to create the feature store and feature groups in Hopsworks.  We also perform extensive exploratory data analysis in this notebook to understand the data.

### 2_heart_training_pipeline

This is where the bulk of the work is performed. We perform oversampling here to improve the predictive performance of the final model. We explore different Sklearn models and perform hyper parameter tuning. We also perform cross validation to ensure that our model is not overfitting. Finally, we save the model to the model registry in Hopsworks.

### 3_heart_feature_pipeline

The heart feature pipeline consists of two parts. The first part consists of the sythetic data pipeline and the second part consists of the user data pipeline. Both pipelines are run daily and add the data to the feature store.

## 4_heart_batch_inference

This notebook is used to perform batch inference on the model. We use the model registry to load the model and then perform inference on the data retrieved from the last 24 hours. We then save the predictions to a monitoring feature store. And save the associated explainability images, confusion matrix and recent predictions as images to be viewed in the monitoring UI.

# UI

1) [Heart-UI](https://huggingface.co/spaces/Potatoasdasdasdasda/Heart-UI)  
2) [Heart-Monitoring](https://huggingface.co/spaces/Potatoasdasdasdasda/Heart-Monitoring)