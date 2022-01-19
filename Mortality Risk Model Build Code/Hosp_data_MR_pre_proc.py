# This script is devoted to the entire preprocessing of the datasets so that they are ready for building ML models.


# Modules
import pandas as pd
import numpy as np
from importlib import reload
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun


# 1) Translate the ICD10 data
# datafun.store_ICD10_translated_covid_data()

# 2) Load the translated datasets with general, sensor and lab data
gen_data, sensor_data, lab_data = datafun.load_translated_covid_data()

# 3) Keep only the columns we are interested in from each dataset by following the MR Implementation thought 1
gen_data2, lab_data2, sensor_data2 = procfun.MR_implementation1_keep_cols(gen_data, lab_data, sensor_data)

# 4) Convert some values and the dates at which measurements occurred into datetime objects
gen_data2, lab_data2, sensor_data2 = procfun.reformat_values_and_dates(gen_data2, lab_data2, sensor_data2)

# 5) Create 2 merged datasets. One that will contain the gen_data, lab_data and the sensor_data with time delay in the
# range [-2, 2] between the first sensor data measurements and the first lab data measurements and one that will not
# contain a time delay between the 2 measurements.
merged_data_td, merged_data_no_td = procfun.create_gen_lab_sensor_datasets(gen_data2, lab_data2, sensor_data2)

# 6) Create a dataset that contains only the gen_data and the sensor_data in order to perform the prediction without
# lab values.
merged_data2 = procfun.create_gen_sensor_data(gen_data2, sensor_data2)

# 7) Replace the column names because 'Leukocytes' and 'Platelet Count' are being misread
loc1 = np.frompyfunc(lambda x: 'Leukocytes' in x, 1, 1)(merged_data_no_td.columns.values)
loc1 = np.where(loc1)[0][0]
loc2 = np.frompyfunc(lambda x: 'Platelet' in x, 1, 1)(merged_data_no_td.columns.values)
loc2 = np.where(loc2)[0][0]

column_names = merged_data_no_td.columns.values.tolist()
column_names[loc1] = 'Leukocytes (10^3/μL)'
column_names[loc2] = 'Platelet Count (10^3/μL)'
merged_data_no_td.columns = column_names
merged_data_td.columns = column_names

# For some reason not filled values at 'Maximum blood pressure value', 'Minimum blood pressure value',
# 'Oxygen saturation value', 'Heart rate value', 'Temperature value' and 'Blood glucose value' columns are 0 instead
# of nan, thus replace 0 values which are illogical with nan manually.
column_names = ['Maximum blood pressure value', 'Minimum blood pressure value', 'Oxygen saturation value',
                'Heart rate value', 'Temperature value', 'Blood glucose value']
for i in column_names:
    merged_data_no_td.loc[merged_data_no_td[merged_data_no_td[i] == 0].index.values, i] = np.nan
    merged_data_td.loc[merged_data_td[merged_data_td[i] == 0].index.values, i] = np.nan
    merged_data2.loc[merged_data2[merged_data2[i] == 0].index.values, i] = np.nan

# 8) Remove columns that have at least 50% of their values not filled
merged_data_no_td = procfun.remove_empty_columns(merged_data_no_td, 0.5)
merged_data_td = procfun.remove_empty_columns(merged_data_td, 0.5)
merged_data2 = procfun.remove_empty_columns(merged_data2, 0.5)

# 9) Remove columns that at least 50% of their values are the same. Exempt certain columns.
accepted_cols = ['COVID diagnosis during admission', 'Reason for discharge as Inpatient',
                 'Date of Admission as Inpatient', 'Sex']
merged_data_no_td = procfun.remove_undiluted_columns(merged_data_no_td, 0.5, accepted_cols)
merged_data_td = procfun.remove_undiluted_columns(merged_data_td, 0.5, accepted_cols)
merged_data2 = procfun.remove_undiluted_columns(merged_data2, 0.5, accepted_cols)

# 10) Replace the outliers in all columns with nan values, except from specific columns.
# Define the columns
column_names = merged_data_no_td.columns.values.tolist()
column_names = np.setdiff1d(column_names, ['ID', 'Age', 'Sex', 'COVID diagnosis during admission',
                                           'Date of Admission as Inpatient', 'Reason for discharge as Inpatient'])

column_names2 = ['Maximum blood pressure value', 'Minimum blood pressure value',
                 'Temperature value', 'Heart rate value', 'Oxygen saturation value']

# Perform the replacements of the outliers with nan
merged_data_no_td.loc[:, column_names] = procfun.replace_outliers_with_nan(merged_data_no_td, column_names)
merged_data_td.loc[:, column_names] = procfun.replace_outliers_with_nan(merged_data_td, column_names)
merged_data2.loc[:, column_names2] = procfun.replace_outliers_with_nan(merged_data2, column_names2)

# 11) Drop rows that have nan values for 'Reason for discharge as Inpatient' and keep only those that have values
# 'Death' or 'Sent Home'
merged_data_no_td = merged_data_no_td.dropna(subset=['Reason for discharge as Inpatient'])
merged_data_no_td = merged_data_no_td[np.isin(merged_data_no_td['Reason for discharge as Inpatient'],
                                              ['Sent Home', 'Death'])]
merged_data_td = merged_data_td.dropna(subset=['Reason for discharge as Inpatient'])
merged_data_td = merged_data_td[np.isin(merged_data_td['Reason for discharge as Inpatient'], ['Sent Home', 'Death'])]
merged_data2 = merged_data2.dropna(subset=['Reason for discharge as Inpatient'])
merged_data2 = merged_data2[np.isin(merged_data2['Reason for discharge as Inpatient'], ['Sent Home', 'Death'])]

# 12) Convert Positive/Negative to 0/1 and Sex from Male/Female to 1/0 and in the 'Reason for discharge as Inpatient'
# consider the 'Death' as 1 (positive class) and the 'Sent Home' as 0 (negative class).
merged_data_no_td.replace({'Male': 1, 'Female': 0, 'Positive': 0, 'Negative': 1, 'Death': 1, 'Sent Home': 0},
                          inplace=True)
merged_data_td.replace({'Male': 1, 'Female': 0, 'Positive': 0, 'Negative': 1, 'Death': 1, 'Sent Home': 0},
                       inplace=True)
merged_data2.replace({'Male': 1, 'Female': 0, 'Positive': 0, 'Negative': 1, 'Death': 1, 'Sent Home': 0},
                     inplace=True)

# 13) Drop rows that correspond to negative covid patients
merged_data_no_td = merged_data_no_td[merged_data_no_td['COVID diagnosis during admission'] == 0]
merged_data_td = merged_data_td[merged_data_td['COVID diagnosis during admission'] == 0]
merged_data2 = merged_data2[merged_data2['COVID diagnosis during admission'] == 0]


# 14) Store or load the data
store = True
if store:
    datafun.store_merged_before_comorbidities(merged_data_no_td.copy(), merged_data_td.copy(), merged_data2.copy())
else:
    merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_before_comorbidities()

# 15) Impute the missing values on all columns using the IterativeImputer from sklearn
'''
merged_data_no_td = procfun.fill_missing_values(merged_data_no_td)
merged_data_td = procfun.fill_missing_values(merged_data_td)
merged_data2 = procfun.fill_missing_values(merged_data2)
'''

# 16) Load the translated ICD10 data
ICD10_data = datafun.load_ICD10_translated_covid_data()

'''
IMPLEMENTATION 1 APPROACH
'''
# 17) For every disease in the ICD10 data, create a column with the disease name and a value of 1 or 0 for every ID that
# has or doesn't have the disease.
comorbidity_dataset = procfun.create_comorbidity_dataset(ICD10_data.copy())

# 18) Drop specific comorbidities that are irrelevant
columns_to_drop = ['Covid-19', 'Fever, unspecified',
                   'Other coronavirus as the cause of diseases classified elsewhere',
                   'Other viral agents as the cause of diseases classified elsewhere',
                   'Delirium due to known physiological condition',
                   'Major depressive disorder, single episode, unspecified',
                   'Other general symptoms and signs', 'Other specified abnormal findings of blood chemistry',
                   'Problems related to living in residential institution',
                   'Contact with and (suspected) exposure to other viral communicable diseases',
                   'Cataract extraction status, right eye', 'Cataract extraction status, left eye',
                   'Presence of intraocular lens', 'Cataract extraction status, unspecified eye']

comorbidity_dataset = comorbidity_dataset.drop(columns=columns_to_drop)

# 19) Find the common IDs of the comorbidity dataset and the 3 merged datasets and merge the comorbditiy dataset with
# the merged datasets
merged_data_no_td_com, merged_data_td_com, merged_data2_com = \
    procfun.create_merged_with_comorbidities(merged_data_no_td, merged_data_td, merged_data2, comorbidity_dataset)

# 20) Store or load the data
store = True
if store:
    datafun.store_merged_after_comorbidities(merged_data_no_td_com.copy(), merged_data_td_com.copy(),
                                             merged_data2_com.copy(), 1)
else:
    merged_data_no_td_com, merged_data_td_com, merged_data2_com = datafun.load_merged_after_comorbidities(1)


'''
IMPLEMENTATION 2 APPROACH - COVIDANALYTICS FORMAT
'''
# 17) Create 4 columns with the general diseases Cardiac arrhythmias, Chronic Kidney Disease,
# Coronary atherosclerosis and other heart disease, Diabetes. For every disease of the aboev that is in the ICD10 data,
# mark the IDs with 1 or 0 for every ID that has or doesn't have the disease.
comorbidity_dataset = procfun.create_comorbidity_dataset2(ICD10_data.copy())

# 19) Find the common IDs of the comorbidity dataset and the 3 merged datasets and merge the comorbditiy dataset with
# the merged datasets
merged_data_no_td_com, merged_data_td_com, merged_data2_com = \
    procfun.create_merged_with_comorbidities(merged_data_no_td, merged_data_td, merged_data2, comorbidity_dataset)

# 20) Store or load the data
store = True
if store:
    datafun.store_merged_after_comorbidities(merged_data_no_td_com.copy(), merged_data_td_com.copy(),
                                             merged_data2_com.copy(), 2)
else:
    merged_data_no_td_com, merged_data_td_com, merged_data2_com = datafun.load_merged_after_comorbidities(2)
