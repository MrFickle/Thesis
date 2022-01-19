import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb


def load_model():
    data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Final Models chosen\Merged data no td\No Severity\All patients'
    filename = r'\xgboost_model_225.pkl'
    clf_xgb = pickle.load(open(data_path + filename, 'rb'))
    return clf_xgb


def sustain_value(key):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]


def show_predict_page():
    # Prediction page title
    st.title("Mortality Risk Calculator")

    # Some text
    st.write("""### Please fill the boxes below in order to make a prediction.""")

    # Load the model
    clf_xgb = load_model()
    # Get the feature names
    feature_names = clf_xgb.feature_names
    # Get the feature types
    feature_types = clf_xgb.feature_types
    # Initialize a dataframe that will contain the inputs
    input_dataframe = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    # Initialize the input keys both as keys and as nan values in the session state
    input_keys = ['input' + str(i) for i in range(0, len(feature_names))]
    for i in range(0, len(feature_names)):
        if input_keys[i] not in st.session_state:
            st.session_state[input_keys[i]] = np.nan

    # Create text boxes to fill
    input_dataframe.loc[0, 'Age'] = st.number_input('Age', value=np.nan, key='input1', on_change=sustain_value('input1'))
    input_dataframe.loc[0, 'Sex'] = st.number_input('Sex', value=np.nan, key='input2', on_change=sustain_value('input2'))
    input_dataframe.loc[0, 'Potasium (mmol/L)'] = st.number_input('Potasium (mmol/L)', value=np.nan, key='input3', on_change=sustain_value('input3'))
    input_dataframe.loc[0, 'Creatinine (mg/dL)'] = st.number_input('Creatinine (mg/dL)', value=np.nan, key='input4', on_change=sustain_value('input4'))
    input_dataframe.loc[0, 'Prothrombin Time (s)'] = st.number_input('Prothrombin Time (s)', value=np.nan, key='input5', on_change=sustain_value('input5'))
    input_dataframe.loc[0, 'Hemoglobin (g/dL)'] = st.number_input('Hemoglobin (g/dL)', value=np.nan, key='input6', on_change=sustain_value('input6'))
    input_dataframe.loc[0, 'Aspartate Aminotransferase (U/L)'] = st.number_input('Aspartate Aminotransferase (U/L)', value=np.nan, key='input7', on_change=sustain_value('input7'))
    input_dataframe.loc[0, 'Blood Glucose (mg/dL)'] = st.number_input('Blood Glucose (mg/dL)', value=np.nan, key='input8', on_change=sustain_value('input8'))
    input_dataframe.loc[0, 'Sodium (mmol/L)'] = st.number_input('Sodium (mmol/L)', value=np.nan, key='input9', on_change=sustain_value('input9'))
    input_dataframe.loc[0, 'C-Reactive Protein (mg/L)'] = st.number_input('C-Reactive Protein (mg/L)', value=np.nan, key='input10', on_change=sustain_value('input10'))
    input_dataframe.loc[0, 'Mean Corpuscular Hemoglobin (pg)'] = st.number_input('Mean Corpuscular Hemoglobin (pg)', value=np.nan, key='input11', on_change=sustain_value('input11'))
    input_dataframe.loc[0, 'Alanine Aminotransferase (U/L)'] = st.number_input('Alanine Aminotransferase (U/L)', value=np.nan, key='input12', on_change=sustain_value('input12'))
    input_dataframe.loc[0, 'Platelet Count (10^3/μL)'] = st.number_input('Platelet Count (10^3/μL)', value=np.nan, key='input13', on_change=sustain_value('input13'))
    input_dataframe.loc[0, 'Leukocytes (10^3/μL)'] = st.number_input('Leukocytes (10^3/μL)', value=np.nan, key='input14', on_change=sustain_value('input14'))
    input_dataframe.loc[0, 'Maximum blood pressure value'] = st.number_input('Maximum blood pressure value', value=np.nan, key='input15', on_change=sustain_value('input15'))
    input_dataframe.loc[0, 'Minimum blood pressure value'] = st.number_input('Minimum blood pressure value', value=np.nan, key='input16', on_change=sustain_value('input16'))
    input_dataframe.loc[0, 'Temperature value'] = st.number_input('Temperature value', value=np.nan, key='input17', on_change=sustain_value('input17'))
    input_dataframe.loc[0, 'Heart rate value'] = st.number_input('Heart rate value', value=np.nan, key='input18', on_change=sustain_value('input18'))
    input_dataframe.loc[0, 'Oxygen saturation value'] = st.number_input('Oxygen saturation value', value=np.nan, key='input19', on_change=sustain_value('input19'))
    input_dataframe.loc[0, 'Cardiac dysrhythmias'] = st.number_input('Cardiac dysrhythmias', value=np.nan, key='input20', on_change=sustain_value('input20'))
    input_dataframe.loc[0, 'Chronic Kidney Disease'] = st.number_input('Chronic Kidney Disease', value=np.nan, key='input21', on_change=sustain_value('input21'))
    input_dataframe.loc[0, 'Coronary atherosclerosis'] = st.number_input('Coronary atherosclerosis', value=np.nan, key='input22', on_change=sustain_value('input22'))
    input_dataframe.loc[0, 'Diabetes'] = st.number_input('Diabetes', value=np.nan, key='input23', on_change=sustain_value('input23'))

    # Display the input_dataframe
    if st.button(label='Show input passed', key='input_data'):
        st.write(input_dataframe.loc[0, :])

    # Make the prediction
    if st.button(label='Make prediction', key='prediction_button'):
        # Create a dummy array for the class
        dummy_y = np.zeros((1, 1))
        # Create a DMatrix using the input given and the dummy class
        dinput = xgb.DMatrix(data=input_dataframe, label=dummy_y)
        # Make the prediction using the xgboost model
        y_pred = clf_xgb.predict(dinput, ntree_limit=clf_xgb.best_iteration + 1)
        # Convert the y_pred value in a integer percentage
        y_pred = np.round(100*y_pred[0]).astype(int)
        # Print the result
        st.subheader(f"Mortality Risk score with input data given: {y_pred} %")




