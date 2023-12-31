# Heart Disease Prediction

Full machine learning system that provides users with the possibility of predicting if they have a risk of heart disease. Furthermore, it gives the administrators the capability of monitoring the state of the model, the data, and the importance of the different features.

This project was done as part of the project course `ID2223: Scalable Machine Learning Project` at `KTH Royal Institute of Technology`.

Links to the user interfaces of the project:
1) [Heart-UI](https://huggingface.co/spaces/Potatoasdasdasdasda/Heart-UI): This is the UI where users can input their data and get a prediction of their risk of heart disease. 
2) [Heart-Monitoring](https://huggingface.co/spaces/Potatoasdasdasdasda/Heart-Monitoring): This is the monitoring UI where the administrators can monitor the state of the model and the data.

## Installation

```bash
conda env create --file environment.yml 
conda activate ID2223_Project
```

## Project Description:

The project mainly used the data provided by the following dataset: [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data). 

We generate new data in two ways:
1) We use a [CTGAN](https://arxiv.org/abs/1907.00503) to generate synthetic data that could be used to enhance the performance of the model. We run a daily feature pipeline to add some extra synthetic data to our feature store.
2) In addition, we allow users to input their own data if they know if they have a heart disease. We store this data in a different feature store, as this information is not static and could change over time. Moreover, we run a daily feature pipeline to add the user data to our main feature store.

This combination of real and synthetic data allows us to have a dynamic dataset that is constantly changing and allows us to monitor and improve the model over time. In addition, it would be very easy to detect if the model is drifting and retrain the model if necessary.

As for the model, we found that a Random Forest Classifier performed the best. We use this model to predict if a patient has a risk of heart disease or not. Where we do a Randomized Search Cross Validation to find the best hyperparameters for the model. While taking into account the fact that the dataset is imbalanced, we use the [SMOTE](https://arxiv.org/abs/1106.1813) algorithm to oversample the minority class.

We also run a daily batch inference job with the new data from the last 24 hours. We save the predictions to a monitoring feature store. Allowing us to monitor the state of the model and the data. While saving the associated explainability images, confusion matrix, and recent predictions as images to be viewed in the monitoring User Interface (UI).

Finally, we created a UI to allow users to input their data and get a prediction of their risk of heart disease. Furthermore, so the application can be used by non-medical professionals, we allow users to input "Unknown" information for certain features that they do not know. We deal with this by imputing the missing values with the most frequent value for that feature, but a more advanced imputation method could be used. This allows users to still receive a prediction of their results even if they do not know all the information, allowing them to seek medical help early.

## Prediction Problem:

Given a patient's information, predict whether the patient has a risk of heart disease or not. This tool could allow the early detection of heart disease and allow patients to seek medical help early before any serious complications arise. Thus saving lives and improving the quality of life of patients.

## Tools Used:

[Scikit-Learn](https://scikit-learn.org/stable/index.html) - ML Models 
[Ydata-synthetic](https://docs.synthetic.ydata.ai/1.3/) - GAN for Synthetic Data
[Hopswork](https://www.hopsworks.ai/) - Model Registry and Feature Store
[Modal](https://modal.com/) - CRON job for Synthetic Data
[Hugging Face](https://huggingface.co/) - UI
[SHAP](https://shap.readthedocs.io/en/latest/) - Explainable AI
[Imbalanced-Learn](https://imbalanced-learn.org/stable/) - Library to deal with Imbalanced Classes 

## Data:

As mentioned above, we used the data provided by the following dataset: [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data). Moreover, more data is added daily through the feature pipelines for both the synthetic data and the user data.

## Methodology and algorithm:

We divided the project into 5 parts:

### 1_heart_eda_and_feature_group

This notebook contains the Explanatory Data Analysis and Feature Group creation. It also contains the code to create the feature store and feature groups in Hopsworks. We also perform extensive exploratory data analysis in this notebook to understand the data.

Furthermore, we train the GAN model, which allows us to generate synthetic data. This required the creation of a special pipeline that inverts the training pipeline. This is because the GAN model could not handle the original data directly and required a training pipeline to perform better. Therefore, the data it generates does not have the same format as the original data and requires the inverse preprocessing steps to be performed on it so that it can be stored in the feature store. Finally, we save the GAN model and the inverse pipeline to the model registry.

### 2_heart_training_pipeline

This is where the bulk of the work is performed. We perform oversampling here to improve the predictive performance of the final model. We explore different Sklearn models and perform hyperparameter tuning through cross-validation to ensure that our model is not overfitting. Finally, we save the model that performs the best to the model registry with the training pipeline that was used to train it.

We found that the Random Forest Classifier performed the best while using the SMOTE algorithm to oversample the minority class while training the model. We also found that the dataset was severely imbalanced and that the model would predict all patients to be healthy, which would result in high accuracy but a useless model. Therefore, we oversampled the minority class to get a more balanced dataset.

### 3_heart_feature_pipeline

The heart feature pipeline consists of two parts. The first part consists of the synthetic data pipeline, and the second part consists of the user data pipeline. Both pipelines are run daily and add the data to the feature store. The synthetic data pipeline uses the GAN model that was trained in the first notebook to generate new data. While the user data pipeline adds the user data from the UI to the feature store.

### 4_heart_batch_inference

This notebook is used to perform batch inference on the model. We use the model registry to load the model and then perform inference on the data retrieved from the last 24 hours. We then save the predictions to a monitoring feature store. And save the associated explainability images, confusion matrix, and recent predictions as images to be viewed in the monitoring UI.

### 5: User Interfaces

We created two user interfaces. One for the users to input their data and get a prediction of their risk of heart disease. And one for the administrators to monitor the state of the model and the data. Both UIs are hosted on Hugging Face Spaces and can be accessed through the links at the top of this page.

## Extra For Excellent 

### 1. Explainable AI

We use the [SHAP](https://shap.readthedocs.io/en/latest/) library to explain the predictions of our model. The full capabilities of SHAP are visible in [playing around](playing_around.ipynb) notebook. However, the library had certain limitations, which resulted in us only using a subset of the features in the final serverless project. Specifically, the library could not generate advanced plots explaining all predictions (we showcase this in playing around). Nontheless, it gives a lot of good information for the user to understand the model.

### 2. Imbalanced Classes

The dataset was severely imbalanced. We used the [Imbalanced-Learn](https://imbalanced-learn.org/stable/) library to deal with this. We used the SMOTE algorithm to oversample the minority class. If we did not perform oversampling, the model would predict all patients to be healthy, which would result in high accuracy but a useless model. Therefore, we oversampled the minority class to get a more balanced dataset.

### 3. Dealing with Missing Values

We allow users to input "Unknown" information for certain features. We deal with this by imputing the missing values with the most frequent value for that feature. This allows users to still receive a prediction of their results even if they do not know all the information.