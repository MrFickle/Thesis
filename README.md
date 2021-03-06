# Mortality risk prediction for COVID-19 patients using XGBoost models

Using demographic and lab test data received from the HM Hospitales in Spain, I built an XGBoost binary classifier using binary logistic regression
that runs on a simple web app using the streamlit module and predicts the mortality risk of a COVID-19 patient. The user has to pass in the appropriate
data as shown in the web app, then click the "Make prediction" button to receive the mortality risk score in a scale of 0-100%. 

The "Mortality Risk Model Build" folder contains all of the main files used to construct the final xgboost model chosen that runs on the web app. It conaints scripts for the construction of the training datasets, data preprocessing, data storage, data plotting/visualization, different approaches of xgboost model creation, hyperparameter tuning, validation processes, xgboost model performance visualization, etc.

The "Mortality Risk Web App" folder contains the scripts required to run the web-app. In order to run the web-app, do the following:
1) Open the "predict_page.py" file and in the load_model() function define the data_path where you've stored the "xgboost_model_225.pkl" file.
2) Go to your IDE's terminal, change directory to the one that contains the web app files and type "streamlit run web_app.py".


WARNING: The specific model is not up to date with the current COVID-19 data and its results should not be taken seriously. A machine learning model
is as good as the data it's trained on.

