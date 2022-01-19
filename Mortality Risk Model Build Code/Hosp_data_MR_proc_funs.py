"""
This script contains functions that are used only for processing data.
"""

# Modules
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import Hosp_data_MR_plot_funs as plotfun


# Create a function that receives the ICD10 dataset, creates a dictionary for every ICD10 code that was found manually
# to be in the top 20 most frequent diseases of every diagnostic column (DIA_PPAL, DIA_02, etc) and decodes the ICD10
# code to disease name. For the diseases found at the top 20, view the google doc below
# https://docs.google.com/document/d/1lTjgRdHLvhnjK7dg65yd9RVsyz9Et7Drwb8qZLcsBQc/edit
# Afterwards, create a dataset with the columns: 'ID', 'Disease1', 'Disease2', etc and mark the specific cells with 1
# or 0 whether that ID has the specific disease or not. In order to place the 1 or 0 view the ICD10 dataset columns
# that contain that information, which are the columns 'POAD_PPAL', 'POAD_02', etc. S and E mean 1 and N means 0.
def create_comorbidity_dataset(ICD10_data):
    # Create the dictionary of the diseases:
    disease_dictionary = {'J12.89': 'Other viral pneumonia', 'U07.1': 'Covid-19',
                          'J18.9': 'Pneumonia, unspecified organism', 'J98.8': 'Other specified respiratory disorders',
                          'J44.0': 'Chronic obstructive pulmonary disease with (acute) lower respiratory infection',
                          'J84.9': 'Interstitial pulmonary disease, unspecified',
                          'J22': 'Unspecified acute lower respiratory infection',
                          'N39.0': 'Urinary tract infection, site not specified', 'J43.9': 'Emphysema, unspecified',
                          'J47.0': 'Bronchiectasis with acute lower respiratory infection',
                          'I26.99': 'Other pulmonary embolism without acute cor pulmonale',
                          'R91.8': 'Other nonspecific abnormal finding of lung field',
                          'J12.81': 'Pneumonia due to SARS-associated coronavirus',
                          'J44.1': 'Chronic obstructive pulmonary disease with (acute) exacerbation',
                          'I11.0': 'Hypertensive heart disease with heart failure',
                          'K52.9': 'Noninfective gastroenteritis and colitis, unspecified',
                          'B34.2': 'Coronavirus infection, unspecified',
                          'J45.901': 'Unspecified asthma with (acute) exacerbation',
                          'A41.9': 'Sepsis, unspecified organism', 'R50.9': 'Fever, unspecified',
                          'B97.29': 'Other coronavirus as the cause of diseases classified elsewhere',
                          'B97.89': 'Other viral agents as the cause of diseases classified elsewhere',
                          'D50.9': 'Iron deficiency anemia, unspecified',
                          'D64.9': 'Anemia, unspecified', 'D69.6': 'Thrombocytopenia, unspecified',
                          'D72.810': 'Lymphocytopenia', 'E03.9': 'Hypothyroidism, unspecified',
                          'E11.65': 'Type 2 diabetes mellitus with hyperglycemia',
                          'E11.9': 'Type 2 diabetes mellitus without complications', 'E66.3': 'Overweight',
                          'E66.9': 'Obesity, unspecified', 'E78.00': 'Pure hypercholesterolemia, unspecified',
                          'E78.5': 'Hyperlipidemia, unspecified',
                          'E79.0': 'Hyperuricemia without signs of inflammatory arthritis and tophaceous disease',
                          'E87.1': 'Hypo-osmolality and hyponatremia',
                          'F05': 'Delirium due to known physiological condition',
                          'F17.210': 'Nicotine dependence, cigarettes, uncomplicated',
                          'F32.9': 'Major depressive disorder, single episode, unspecified',
                          'G47.33': 'Obstructive sleep apnea (adult) (pediatric)',
                          'G62.81': 'Critical illness polyneuropathy', 'I10': 'Essential (primary) hypertension',
                          'I12.9': 'Hypertensive chronic kidney disease with stage 1 through stage 4 chronic kidney '
                                   'disease, or unspecified chronic kidney disease',
                          'I25.9': 'Chronic ischemic heart disease, unspecified',
                          'I48.2': 'Chronic atrial fibrillation',
                          'I48.91': 'Unspecified atrial fibrillation', 'I50.9': 'Heart failure, unspecified',
                          'I67.82': 'Cerebral ischemia', 'J80': 'Acute respiratory distress syndrome',
                          'J96.00': 'Acute respiratory failure, unspecified whether with hypoxia or hypercapnia',
                          'J96.01': 'Acute respiratory failure with hypoxia',
                          'J96.90': 'Respiratory failure, unspecified, unspecified whether with hypoxia or hypercapnia',
                          'J96.91': 'Respiratory failure, unspecified with hypoxia',
                          'K44.9': 'Diaphragmatic hernia without obstruction or gangrene',
                          'K57.30': 'Diverticulosis of large intestine without perforation or abscess without bleeding',
                          'N17.9': 'Acute kidney failure, unspecified', 'N18.9': 'Chronic kidney disease, unspecified',
                          'N40.0': 'Benign prostatic hyperplasia without lower urinary tract symptoms',
                          'R09.02': 'Hypoxemia', 'R68.89': 'Other general symptoms and signs',
                          'R74.0': 'Nonspecific elevation of levels of transaminase and lactic acid dehydrogenase '
                                   '[LDH]',
                          'R79.89': 'Other specified abnormal findings of blood chemistry',
                          'T38.0X5A': 'Adverse effect of glucocorticoids and synthetic analogues, initial encounter',
                          'Y84.8': 'Other medical procedures as the cause of abnormal reaction of the patient, or of '
                                   'later complication, without mention of misadventure at the time of the procedure',
                          'Z20.828': 'Contact with and (suspected) exposure to other viral communicable diseases',
                          'Z59.3': 'Problems related to living in residential institution',
                          'Z79.01': 'Long term (current) use of anticoagulants',
                          'Z79.02': 'Long term (current) use of antithrombotics/antiplatelets',
                          'Z79.4': 'Long term (current) use of insulin',
                          'Z79.52': 'Long term (current) use of systemic steroids',
                          'Z79.82': 'Long term (current) use of aspirin',
                          'Z79.84': 'Long term (current) use of oral hypoglycemic drugs',
                          'Z79.891': 'Long term (current) use of opiate analgesic',
                          'Z79.899': 'Other long term (current) drug therapy',
                          'Z85.038': 'Personal history of other malignant neoplasm of large intestine',
                          'Z85.46': 'Personal history of malignant neoplasm of prostate',
                          'Z86.19': 'Personal history of other infectious and parasitic diseases',
                          'Z86.711': 'Personal history of pulmonary embolism',
                          'Z86.73': 'Personal history of transient ischemic attack (TIA), and cerebral infarction '
                                    'without residual deficits',
                          'Z87.891': 'Personal history of nicotine dependence', 'Z88.0': 'Allergy status to penicillin',
                          'Z88.1': 'Allergy status to other antibiotic agents',
                          'Z88.6': 'Allergy status to analgesic agent',
                          'Z88.8': 'Allergy status to other drugs, medicaments and biological substances',
                          'Z90.49': 'Acquired absence of other specified parts of digestive tract',
                          'Z90.710': 'Acquired absence of both cervix and uterus',
                          'Z90.722': 'Acquired absence of ovaries, bilateral',
                          'Z90.79': 'Acquired absence of other genital organ(s)',
                          'Z91.041': 'Radiographic dye allergy status',
                          'Z92.21': 'Personal history of antineoplastic chemotherapy',
                          'Z92.3': 'Personal history of irradiation',
                          'Z95.0': 'Presence of cardiac pacemaker',
                          'Z95.5': 'Presence of coronary angioplasty implant and graft',
                          'Z96.1': 'Presence of intraocular lens', 'Z98.41': 'Cataract extraction status, right eye',
                          'Z98.42': 'Cataract extraction status, left eye',
                          'Z98.49': 'Cataract extraction status, unspecified eye',
                          'Z99.81': 'Dependence on supplemental oxygen',
                          'Z99.89': 'Dependence on other enabling machines and devices'}
    # Get the dict keys
    dict_keys = list(disease_dictionary.keys())

    # Get the IDs of the ICD10 dataset
    ids = ICD10_data['ID'].values
    # Initialize the comorbidity dataset
    cols = ['ID'] + list(disease_dictionary.values())
    comorbidity_dataset = pd.DataFrame(columns=cols, index=np.arange(0, len(ids)))
    comorbidity_dataset['ID'] = ids
    comorbidity_dataset.iloc[:, 1:] = 0

    # Now time to place values into the cells
    # Define the disease diagnostic column names, as well as the column names of whether they have or not the disease
    # during admission (Present On Admission)
    dia_list = ['DIA_PPAL']
    poad_list = ['POAD_PPAL']
    temp1 = 'DIA_'
    temp2 = 'POAD_'
    for j in range(2, 20):
        if j <= 9:
            dia_list.append(temp1 + '0' + str(j))
            poad_list.append(temp2 + '0' + str(j))
        else:
            dia_list.append(temp1 + str(j))
            poad_list.append(temp2 + str(j))

    # For every ID get its row, find the POAD columns for which it has 'S' or 'E' and then go to the respective DIA
    # column and use the disease dictionary to mark the corresponding cell in the comorbidity_dataset.
    for i in range(0, len(ids)):
        # Get the id
        id = ids[i]
        # Get the row
        row = ICD10_data[ICD10_data['ID'] == id]
        # Get the POAD columns that have S or E values
        poad_columns = row[poad_list]
        poad_columns = poad_columns[(poad_columns == 'S') | (poad_columns == 'E')]
        poad_columns = poad_columns.columns.values
        # Keep the index part (after the _ ) of the column name to use for the DIA columns
        poad_index = [j.split('_')[1] for j in poad_columns]
        # Now find the ICD10 disease keys of the row
        row_keys = row.loc[:, ['DIA_' + j for j in poad_index]].values.tolist()[0]
        # Keep only the ones that belong in the dictionary keys
        common_keys = np.intersect1d(row_keys, dict_keys).tolist()
        # Now go to the corresponding cell in the comorbidity_dataset and mark it with 1.
        id_index = comorbidity_dataset[comorbidity_dataset['ID'] == id].index.values
        comorbidity_dataset.loc[id_index, [disease_dictionary[j] for j in common_keys]] = 1

    # Return the comorbidity dataset
    return comorbidity_dataset


# Create a function that receives the ICD10 dataset as well as number of top x frequent diseases to keep and it finds
# the unique top x diseases from all diagnostic columns:
def find_top_x_diseases(ICD10_data, top_x):
    # Disease diagnostic column names
    dia_list = ['DIA_PPAL']
    temp = 'DIA_'
    for j in range(2, 20):
        if j <= 9:
            dia_list.append(temp + '0' + str(j))
        else:
            dia_list.append(temp + str(j))

    # Get the top x values of all columns and place them into 1 column
    top_x_values = list()
    for i in dia_list:
        value_freq = ICD10_data[i].value_counts()
        top_x_values = top_x_values + value_freq.index.values[0:top_x].tolist()

    # Keep only the unique values
    top_x_values = np.unique(top_x_values)

    # Return the values
    return top_x_values


# Create a function that receives the gen_data, lab_data and sensor_data datasets and keeps only certain columns for
# each of them, based on the IR Implementation thought 1. Specifically:
# 1) gen_data: ['ID', 'Age', 'Sex', 'COVID diagnosis during admission']
# 2) lab_data: ['ID', 'Lab ID', 'Test Date', 'Lab Test Name', 'Test Value', 'Measurement Units', 'Ref Values']
# 3) sensor_data: ['ID', 'Constant record date', 'Maximum blood pressure value', 'Minimum blood pressure value',
#                 'Temperature value', 'Heart rate value', 'Oxygen saturation value', 'Blood glucose value']
def MR_implementation1_keep_cols(gen_data, lab_data, sensor_data):
    cols_to_keep = ['ID', 'Age', 'Sex', 'COVID diagnosis during admission', 'Date of Admission as Inpatient',
                    'Reason for discharge as Inpatient']
    gen_data2 = gen_data[cols_to_keep].copy()

    # lab_data
    cols_to_keep = ['ID', 'Lab ID', 'Test Date', 'Lab Test Name', 'Test Value', 'Measurement Units', 'Ref Values']
    lab_data2 = lab_data[cols_to_keep].copy()

    # sensor_data
    cols_to_keep = ['ID', 'Constant record date', 'Maximum blood pressure value', 'Minimum blood pressure value',
                    'Temperature value', 'Heart rate value', 'Oxygen saturation value', 'Blood glucose value']
    sensor_data2 = sensor_data[cols_to_keep].copy()

    # Return the data
    return gen_data2, lab_data2, sensor_data2


# Create a function that receives the gen_data, lab_data and the sensor_data and replaces certain values that need
# reformatting as well as it converts the date strings into datetime objects.
def reformat_values_and_dates(gen_data, lab_data, sensor_data):
    # In the gen_data convert the 'Suspected Covid' value to 'Negative' after cross validating the same IDs of the
    # 19/04/2021 datasets with the 20/07/2020 datasets for these specific values, revealing that
    # 'Suspected Covid' @ 19/04/21 = 'Negative' @ 20/07/2020. Also, change the 'Confirmed Covid' value to 'Positive'.
    gen_data.replace({'Confirmed Covid': 'Positive', 'Suspected Covid': 'Negative'}, inplace=True)

    # For the gen_data replace '/' in dates with '-'.
    gen_data['Date of Admission as Inpatient'] = gen_data['Date of Admission as Inpatient'].str.replace('/', '-')
    # Convert the dates into datetime objects
    gen_data['Date of Admission as Inpatient'] = pd.to_datetime(gen_data['Date of Admission as Inpatient'],
                                                                format="%Y-%m-%d")
    # For the lab_data replace '/' in dates with '-'.
    lab_data['Test Date'] = lab_data['Test Date'].str.replace('/', '-')

    # Convert the dates at which measurements occurred into datetime objects
    # lab_data
    lab_data['Test Date'] = pd.to_datetime(lab_data['Test Date'], format="%d-%m-%Y")

    # sensor_data
    sensor_data['Constant record date'] = pd.to_datetime(sensor_data['Constant record date'], format="%Y-%m-%d")

    # Return the reformatted dataframes
    return gen_data, lab_data, sensor_data


# Create a function that receives the gen_data, lab_data and sensor_data and creates 2 merged datasets out of these
# datasets. They both contain all 3 datasets, one with time delay between the first sensor and the first lab
# measurements and one without. The first date of testing must be after the date of admission.
def create_gen_lab_sensor_datasets(gen_data, lab_data, sensor_data):
    # MERGED DATASET 1: CONTAINS GEN_DATA, LAB_DATA, SENSOR_DATA
    # Find the common IDs across the 3 datasets
    common_IDs1 = np.intersect1d(gen_data['ID'].values, lab_data['ID'].values)
    common_IDs1 = np.intersect1d(common_IDs1, sensor_data['ID'].values)

    # Initialize the merged dataset
    merged_data1 = gen_data[gen_data['ID'].isin(common_IDs1)].copy()

    # Add columns from the sensor and lab tests to the respective merged datasets
    merged_data1, lab_dict = create_lab_sensor_cols(merged_data1, lab_data)

    # Create a dataframe for every common ID in sensor and lab data and create columns with the earliest and latest
    # dates of measurement to see if there is an overlap
    dates = create_dates_data(lab_data, sensor_data, gen_data)
    # Check for overlap between lab data and sensor data for the merged_data only.
    dates = check_date_overlap(dates.copy())
    # Keep only the cases with overlap
    dates = dates[dates['Overlap'] == 1]

    # Plot step: Plot the ECDF for the time delays to see how the values are distributed in the sample

    #store_path = r'H:\Documents\Coding\Projects\Thesis\Plots\Mortality Risk' + \
    #             r'\MR_implementation1_lab_sensor_time_delay_ecdf.png'
    #plotfun.plot_ECDF(dates['Time Delay'].values, data_path=store_path)


    # Based on the above plot, we can pretty much see that at least 90% of the values of the time delay lie in the range
    # [-2, 2]. Thus we are going to use this range only in order to get our overlapped lab and sensor data.
    dates = dates[dates['Time Delay'].between(-2, 2)]

    # Define the date that is going to be used to get both the lab and sensor data for every ID
    dates = define_use_date(dates)
    # For the Use Date created for every ID check whether it is available both for the sensor and the lab data.
    dates = check_use_date_IDs(lab_data, sensor_data, dates)
    # Create one dataset from the marked IDs and another one for all of the IDs separately for the lab and the sensor
    # data, thus creating 2 types of merged_data to be tested and compared in terms of performance later.
    # Split the dates dataframe into one with marked IDs and one without
    dates_no_td = dates[dates['Mark ID'] == 1].copy()
    dates_td = dates[dates['Mark ID'] == 0].copy()

    # Use the above dataframes along with lab_data2 and sensor_data2 to create the lab and sensor data using specific
    # dates of measurement. Use Date for the marked, and first lab/sensor dates for the unmarked.
    lab_data_no_td, sensor_data_no_td, lab_data_td, sensor_data_td = use_date_lab_sensor(dates_td, dates_no_td,
                                                                                         lab_data, sensor_data)
    # Create 2 copies of the merged_data one which will contain the lab and sensor data with no time delay and one with
    # and without. Get the respective IDs and create the copies.
    no_td_IDs = lab_data_no_td['ID'].values
    td_IDs = lab_data_td['ID'].values
    merged_data_no_td = merged_data1[merged_data1['ID'].isin(no_td_IDs)]
    merged_data_td = merged_data1[merged_data1['ID'].isin(td_IDs)]

    # Pass the lab values in both copies of the merged_data
    merged_data_no_td = pass_lab_values(lab_data_no_td, merged_data_no_td.copy(), lab_dict)
    merged_data_td = pass_lab_values(lab_data_td, merged_data_td.copy(), lab_dict)

    # Fix the sensor data because they contain duplicate rows for every ID both for the copies of merged_data and for
    # the merged_data2
    sensor_data_no_td = fix_sensor_values(sensor_data_no_td.copy())
    sensor_data_td = fix_sensor_values(sensor_data_td.copy())
    # Pass the sensor values in both copies of the merged_data and in merged_data2
    merged_data_no_td = pass_sensor_values(sensor_data_no_td.copy(), merged_data_no_td.copy())
    merged_data_td = pass_sensor_values(sensor_data_td.copy(), merged_data_td.copy())
    # Drop the 'Blood glucose value' column which originated from the sensor data since its empty and we already have
    # lab values for it
    # merged_data_no_td = merged_data_no_td.drop(columns='Blood glucose value')
    # merged_data_td = merged_data_td.drop(columns='Blood glucose value')

    # Return the 2 merged datasets
    return merged_data_td, merged_data_no_td


# Create a function that places the lab test names and the sensor tests as columns in the merged dataset and initializes
# them with NaN values and then returns the new merged dataset as well as a dictionary that has as keys the names of the
# tests as found in the lab dataset and as values the names of the respective columns in the merged dataset.
def create_lab_sensor_cols(merged_data, lab_data):
    # Create the column names from lab_data and initialize them with NaN
    lab_tests = list(set(lab_data['Lab Test Name']))
    m_units = list()
    mu_col_index = lab_data.columns.get_loc('Measurement Units')
    for x in lab_tests:
        temp = lab_data[lab_data['Lab Test Name'] == x].copy().iloc[0, mu_col_index]
        m_units.append(temp)

    lab_dict = {}
    new_col_names = list()
    for i in range(0, len(lab_tests)):
        temp = lab_tests[i] + ' (' + str(m_units[i]) + ')'
        lab_dict[lab_tests[i]] = temp
        new_col_names.append(temp)

    # Add the new column names from lab_data2 into merged_data and place NaN values
    merged_data[new_col_names] = float('NaN')

    # Add the new column names from sensor_data2 and place NaN values. For this part, I should at some point search for
    # the measurement units of these tests because they are not given in sensor_data2.
    sensor_col_names = ['Maximum blood pressure value', 'Minimum blood pressure value',
                     'Temperature value', 'Heart rate value', 'Oxygen saturation value', 'Blood glucose value']

    merged_data[sensor_col_names] = float('NaN')
    # Return the merged_data
    return merged_data, lab_dict


# Create a function that creates a dataframe that contains the common IDs for the lab and sensor data and creates
# columns that contain the earliest and latest dates of test performed both for lab and sensor data.
def create_dates_data(lab_data2, sensor_data2, gen_data2):
    # Find earliest dates for lab_data2
    early_lab = earliest_dates('ID', 'Test Date', lab_data2.copy(), gen_data2.copy())
    # Remove duplicate IDs
    early_lab = early_lab.drop_duplicates('ID', keep='first')

    # Find latest dates for lab_data2
    late_lab = latest_dates('ID', 'Test Date', lab_data2.copy(), gen_data2.copy())
    # Remove duplicate IDs
    late_lab = late_lab.drop_duplicates('ID', keep='first')

    # Find earliest dates for sensor_data2
    early_sensor = earliest_dates('ID', 'Constant record date', sensor_data2.copy(), gen_data2.copy())
    # Remove duplicate IDs
    early_sensor = early_sensor.drop_duplicates('ID', keep='first')

    # Find latest dates for sensor_data2
    late_sensor = latest_dates('ID', 'Constant record date', sensor_data2.copy(), gen_data2.copy())
    # Remove duplicate IDs
    late_sensor = late_sensor.drop_duplicates('ID', keep='first')

    # Find the new common IDs across all date dataframes
    new_common_IDs = np.intersect1d(early_lab['ID'].values, late_lab['ID'].values)
    new_common_IDs = np.intersect1d(new_common_IDs, early_sensor['ID'].values)
    new_common_IDs = np.intersect1d(new_common_IDs, late_sensor['ID'].values)

    # Keep IDs belonging in new_common_IDs
    early_lab = early_lab[early_lab['ID'].isin(new_common_IDs)]
    # Sort values in ascending order based on ID
    early_lab = early_lab.sort_values(by='ID', ascending=True)
    # Keep IDs belonging in new_common_IDs
    late_lab = late_lab[late_lab['ID'].isin(new_common_IDs)]
    # Sort values in ascending order based on ID
    late_lab = late_lab.sort_values(by='ID', ascending=True)
    # Keep IDs belonging in new_common_IDs
    early_sensor = early_sensor[early_sensor['ID'].isin(new_common_IDs)]
    # Sort values in ascending order based on ID
    early_sensor = early_sensor.sort_values(by='ID', ascending=True)
    # Keep IDs belonging in new_common_IDs
    late_sensor = late_sensor[late_sensor['ID'].isin(new_common_IDs)]
    # Sort values in ascending order based on ID
    late_sensor = late_sensor.sort_values(by='ID', ascending=True)

    # Initialize new dataframe containing dates
    dates = pd.DataFrame(data={'ID': new_common_IDs})
    # Sort values in ascending order based on ID
    dates = dates.sort_values(by='ID', ascending=True)

    # Add the earliest dates and latest dates in the dates dataframe
    dates['Early Lab'] = early_lab['Test Date'].values
    dates['Late Lab'] = late_lab['Test Date'].values
    dates['Early Sensor'] = early_sensor['Constant record date'].values
    dates['Late Sensor'] = late_sensor['Constant record date'].values

    return dates


# This function receives the dates dataframe in order to find overlap zones between lab and sensor measurements, so that
# we have values from both datasets on the same day when training our model. It returns the original dataframe, along
# with a new column which contains information whether there is overlap or not as well as a column showing the time
# delay between the first lab test and the first sensor test. Positive values indicate the number of days for the
# first sensor test after the first lab test. Negative values indicate the number of days the first sensor test was
# done prior to the first lab test.
def check_date_overlap(dates):
    # Col1: 'ID', Col2: 'Early Lab', Col3: 'Late Lab', Col4: 'Early Sensor', Col5: 'Late Sensor'

    # Get number of IDs
    l = dates.shape[0]
    # Initialize overlap values
    overlap = np.zeros(l)
    # Initialize time delay between first lab and first sensor test
    time_d = np.zeros(l)
    # Get dates columns
    early_lab = dates['Early Lab'].values
    late_lab = dates['Late Lab'].values
    early_sensor = dates['Early Sensor'].values
    late_sensor = dates['Late Sensor'].values

    # Check for overlap for every ID
    for i in range(0, l):
        # Case 1: Sensor data first measurement occurred after first lab measurement but before the last lab measurement
        if early_sensor[i] >= early_lab[i] and early_sensor[i] <= late_lab[i]:
            overlap[i] = True

        # Case 2: Lab data first measurement occurred after first sensor measurement but before the last sensor
        # measurement
        elif early_sensor[i] <= early_lab[i] and early_lab[i] <= late_sensor[i]:
            overlap[i] = True

        # Case 3: Sensor data finished measurements before lab measurements began, or vice versa
        elif late_sensor[i] <= early_lab[i] or early_sensor[i] >= late_lab[i]:
            overlap[i] = False
        else:
            overlap[i] = 'Nani?!'

        if overlap[i]:
            temp = early_sensor[i] - early_lab[i]
            temp = temp.astype('timedelta64[D]')
            temp = temp.item().days
            time_d[i] = temp
        else:
            time_d[i] = float('NaN')

    # Create overlap column
    dates['Overlap'] = overlap
    # Create time delay column between first sensor test and first lab test
    dates['Time Delay'] = time_d

    return dates


# Create a function that receives the dates dataframe after its filtered using the Time Delay values and defines the
# date that will be used to get the lab and sensor data. It returns the original dataframe, while also creating a new
# column called 'Use Date'.
def define_use_date(dates):
    # Number of IDs
    l = dates.shape[0]
    # Early lab, early sensor and time delay values
    early_lab = dates['Early Lab'].values
    early_sensor = dates['Early Sensor'].values
    time_d = dates['Time Delay'].values
    # Initialize use date vector
    use_date = list()
    # For every ID using the first lab date as reference, get the date to use by adding the time delay value. If the
    # time delay is negative, then the use date will be the date of the early lab date. On the other hand, if the
    # time delay is positive, then the use date will be the date of the early sensor date. A different approach would be
    # to do the following: 1) If the time delay is positive then add that value to the early lab date to find the use
    # date
    #                      2) If the time delay is negative then add the absolute value to the early sensor date to find
    # the use date
    for i in range(0, l):
        if time_d[i] >= 0:
            use_date.append(early_sensor[i])
        else:
            use_date.append(early_lab[i])

    # Add the use_date in a new column
    dates['Use Date'] = use_date
    # Return the dates dataframe
    return dates


# Create a function that returns the dates, with an added column that marks the IDs for which both the sensor and the
# lab data contain tests on the Use Date defined previously.
def check_use_date_IDs(lab_data2, sensor_data2, dates):
    # Get IDs
    IDs = dates['ID'].values
    # Initialize mark ID vector
    mark_ID = np.zeros(len(IDs))
    # Get Use Date vector
    use_date = dates['Use Date'].values
    # For every ID check if the use date is available for both sensor and lab data
    for i in range(0, len(IDs)):
        temp_lab = lab_data2[lab_data2['ID'] == IDs[i]].copy()
        temp_sensor = sensor_data2[sensor_data2['ID'] == IDs[i]].copy()
        # Check if the use date belongs in temp_lab and temp_sensor
        if not(temp_lab[temp_lab['Test Date'] == use_date[i]].empty):
            if not(temp_sensor[temp_sensor['Constant record date'] == use_date[i]].empty):
                mark_ID[i] = True
            else:
                mark_ID[i] = False
        else:
            mark_ID[i] = False

    # Add mark ID column to dates dataframe
    dates['Mark ID'] = mark_ID

    # Return the dates dataframe
    return dates


# Create a function that receives the dates_no_td and the dates_td dataframe, as well as the lab_data2 and the
# sensor_data2 datasets and returns 4 datasets lab_data_no_td, sensor_data_no_td, lab_data_td, sensor_data_td. The first
# 2 contain the IDs for which there is no time delay between the first lab and the first measurement, while keeping the
# measurements that occurred only in the use date. The latter 2 contain the IDs for which there is a time delay between
# the first lab and the first measurement in the range [-2, 2] while keeping the measurements that occurred only in the
# respective use dates, which are not the same for the lab and the sensor data. However, for every ID only one date will
# be used to gather the lab and sensor data in each of the 2 separate datasets.
def use_date_lab_sensor(dates_td, dates_no_td, lab_data2, sensor_data2):
    # No time delay datasets
    # Get IDs for which there is no time delay
    no_td_IDs = dates_no_td['ID'].values
    # Create the lab_data_no_td and the sensor_data_no_td
    lab_data_no_td = lab_data2[lab_data2['ID'].isin(no_td_IDs)].copy()
    sensor_data_no_td = sensor_data2[sensor_data2['ID'].isin(no_td_IDs)].copy()
    # Keep only measurements for the respective use dates
    use_date_no_td = dates_no_td['Use Date'].values
    # Initialize 2 arrays that will contain the row indices of the lab_data_no_td and the sensor_data_no_td to keep
    rows_to_keep_lab = np.empty(0)
    rows_to_keep_sensor = np.empty(0)
    for i in range(0, len(no_td_IDs)):
        # Get lab and sensor data for a specific ID
        temp_lab = lab_data_no_td[lab_data_no_td['ID'] == no_td_IDs[i]].copy()
        temp_sensor = sensor_data_no_td[sensor_data_no_td['ID'] == no_td_IDs[i]].copy()
        # Keep only the rows that contain the use date
        temp_lab = temp_lab[temp_lab['Test Date'] == use_date_no_td[i]]
        temp_sensor = temp_sensor[temp_sensor['Constant record date'] == use_date_no_td[i]]
        # Append the row indices of the rows to keep
        rows_to_keep_lab = np.append(rows_to_keep_lab, temp_lab.index)
        rows_to_keep_sensor = np.append(rows_to_keep_sensor, temp_sensor.index)

    # Keep only the rows defined previously separately for the sensor and the lab data
    lab_data_no_td = lab_data_no_td[lab_data_no_td.index.isin(rows_to_keep_lab)]
    sensor_data_no_td = sensor_data_no_td[sensor_data_no_td.index.isin(rows_to_keep_sensor)]

    # Time delay datasets
    # Get IDs for which there is a time delay
    td_IDs = dates_td['ID'].values
    # Create the lab_data_td and the sensor_data_td
    lab_data_td = lab_data2[lab_data2['ID'].isin(td_IDs)].copy()
    sensor_data_td = sensor_data2[sensor_data2['ID'].isin(td_IDs)].copy()
    # The use date for the lab data will be the one found in Early Lab, while the use date for the sensor data will be
    # the one found in Early Sensor.
    use_date_lab = dates_td['Early Lab'].values
    use_date_sensor = dates_td['Early Sensor'].values
    # Initialize 2 arrays that will contain the row indices of the lab_data_td and the sensor_data_td to keep
    rows_to_keep_lab = np.empty(0)
    rows_to_keep_sensor = np.empty(0)
    for i in range(0, len(td_IDs)):
        # Get lab and sensor data for a specific ID
        temp_lab = lab_data_td[lab_data_td['ID'] == td_IDs[i]].copy()
        temp_sensor = sensor_data_td[sensor_data_td['ID'] == td_IDs[i]].copy()
        # Keep only the rows that contain the respective use date
        temp_lab = temp_lab[temp_lab['Test Date'] == use_date_lab[i]]
        temp_sensor = temp_sensor[temp_sensor['Constant record date'] == use_date_sensor[i]]
        # Append the row indices of the rows to keep
        rows_to_keep_lab = np.append(rows_to_keep_lab, temp_lab.index)
        rows_to_keep_sensor = np.append(rows_to_keep_sensor, temp_sensor.index)

    # Keep only the rows defined previously separately for the sensor and the lab data
    lab_data_td = lab_data_td[lab_data_td.index.isin(rows_to_keep_lab)]
    sensor_data_td = sensor_data_td[sensor_data_td.index.isin(rows_to_keep_sensor)]

    # Merge the lab_data_td with the lab_data_no_td to get the complete lab dataset with time delay
    lab_data_td = lab_data_td.append(lab_data_no_td)
    # Merge the sensor_data_td with the sensor_data_no_td to get the complete sensor dataset with time delay
    sensor_data_td = sensor_data_td.append(sensor_data_no_td)

    # Return the final datasets
    return lab_data_no_td, sensor_data_no_td, lab_data_td, sensor_data_td


# Create a function that passes the lab values from a lab dataset into a merged_data dataset using the lab dictionary
# produced from the function create_lab_sensor_cols.
def pass_lab_values(lab_data, merged_data, lab_dict):
    # All IDs
    unique_ID = np.unique(merged_data['ID'].values)
    # IDs with no lab values
    no_data_IDs = list()

    # Get values for every test for all the IDs
    for ID in unique_ID:
        # Get row index for the ID in merged_data
        row_index = merged_data[merged_data['ID'] == ID].index
        if ID in lab_data['ID'].values:
            # Get lab tests recorded for the ID
            block = lab_data[lab_data['ID'] == ID].copy()
            for test in lab_dict.keys():
                if test in block['Lab Test Name'].values:
                    # For every test get the value and pass it to the corresponding cell in merged_data
                    lab_row = block[block['Lab Test Name'] == test].copy()
                    # In case of multiple recordings of the same test on the same day, keep the first one
                    if lab_row.shape[0] > 1:
                        lab_row = lab_row.iloc[0, :]
                        lab_row = lab_row.to_frame()
                        lab_row = lab_row.T
                    # Get value
                    value = lab_row['Test Value'].values
                    # Convert to float, skip buggy values
                    try:
                        value = float(value)
                    except ValueError:
                        continue

                    # Pass value to the corresponding merged_data cell
                    col_index = lab_dict[test]
                    merged_data.loc[row_index, col_index] = value

        # If the specific ID has no lab values, then remove it
        else:
            no_data_IDs.append(ID)

    # Remove IDs with no data
    merged_data = merged_data[~merged_data['ID'].isin(no_data_IDs)]

    return merged_data


# Create a function that combines sensor data rows into 1 when the ID and the date is the same, but there are multiple
# rows. In case of multiple measurements of the same type on the same day, the mean is placed in the combined row.
def fix_sensor_values(sensor_data):
    # First transform the 'Temperature value' values into floats instead of strings.
    sensor_data['Temperature value'] = sensor_data['Temperature value'].str.replace(',', '.').astype(float)
    # Get IDs
    IDs = sensor_data['ID'].values
    # For every ID find its respective block in the sensor data
    for i in IDs:
        block = sensor_data[sensor_data['ID'] == i].copy()
        # If the block contains one row, then go to the next ID
        if block.shape[0] == 1:
            continue
        # Otherwise, merge rows by creating one row for this ID, where the column values are the mean of the existent
        # column values of the separate rows. If a row has for a specific column a 0 value, then it will not be used
        # in calculating the mean for the specific column.
        else:
            # Keep the row indices of the block
            row_index = block.index
            # Don't use the columns 'ID' and 'Constant record date' since we don't need them for this part
            for col in block.columns[~block.columns.isin(['ID', 'Constant record date'])]:
                col_value = block[col].values
                col_value = col_value[col_value != 0]
                if len(col_value) == 0:
                    col_value = 0
                else:
                    col_value = float(np.mean(col_value))
                # Replace the value of that column for all rows with the col_value. The redundant ones will be dropped
                # later
                sensor_data.loc[row_index, col] = col_value

    # Drop ID duplicate rows
    sensor_data = sensor_data.drop_duplicates(subset='ID', keep='first')

    return sensor_data


# Create a function that passes the sensor values from a sensor dataset into a merged_data dataset.
def pass_sensor_values(sensor_data, merged_data):
    # Get IDs
    IDs = merged_data['ID'].values
    # For every ID add the sensor values to the merged_data
    for i in IDs:
        block_sensor = sensor_data[sensor_data['ID'] == i].copy()
        row_index = merged_data[merged_data['ID'] == i].index
        if block_sensor.shape[0] == 1:
            for col in block_sensor.columns[~block_sensor.columns.isin(['ID', 'Constant record date'])]:
                merged_data.loc[row_index, col] = block_sensor[col].values
        else:
            continue

    return merged_data


# This function receives the name of the column "ID_col_name", which it uses as an ID to iterate all rows
# of the dataframe df, and keeps only the ones that have the same minimum date value in the column "date_col_name"
# for the corresponding ID value.
def earliest_dates(ID_col_name, date_col_name, df, gen_data2):
    new_df = pd.DataFrame()
    # Find the unique IDs
    IDs = set(df[ID_col_name])
    # Iterate through the rows for each ID and keep the ones with minimum dates
    for i in IDs:
        # Get the date of admission for the specific ID
        admission_date = gen_data2[gen_data2['ID'] == i]
        admission_date = admission_date.loc[admission_date.index.values[0], 'Date of Admission as Inpatient']
        # Keep only the rows with ID = i
        ID_i = df[df[ID_col_name] == i].copy()
        # From those rows, keep only the ones that are >= the admission date
        ID_i = ID_i[ID_i[date_col_name] >= admission_date]
        # If no such date exists, then skip this ID
        if ID_i.empty:
            continue
        else:
            # Find the minimum date for ID = i
            min_date = min(ID_i[date_col_name])
            # Keep only the rows of ID = i with the minimum date
            ID_i = ID_i[ID_i[date_col_name] == min_date]
            new_df = new_df.append(ID_i, ignore_index=True)

    return new_df


# This function receives the name of the column "ID_col_name", which it uses as an ID to iterate all rows
# of the dataframe df, and keeps only the ones that have the same maximum date value in the column "date_col_name"
# for the corresponding ID value.
def latest_dates(ID_col_name, date_col_name, df, gen_data2):
    new_df = pd.DataFrame()
    # Find the unique IDs
    IDs = set(df[ID_col_name])
    # Iterate through the rows for each ID and keep the ones with minimum dates
    for i in IDs:
        # Get the date of admission for the specific ID
        admission_date = gen_data2[gen_data2['ID'] == i]
        admission_date = admission_date.loc[admission_date.index.values[0], 'Date of Admission as Inpatient']
        # Keep only the rows with ID = i
        ID_i = df[df[ID_col_name] == i].copy()
        # From those rows, keep only the ones that are >= the admission date
        ID_i = ID_i[ID_i[date_col_name] >= admission_date]
        # If no such date exists, then skip this ID
        if ID_i.empty:
            continue
        else:
            # Find the minimum date for ID = i
            max_date = max(ID_i[date_col_name])
            # Keep only the rows of ID = i with the minimum date
            ID_i = ID_i[ID_i[date_col_name] == max_date]
            new_df = new_df.append(ID_i, ignore_index=True)

    return new_df


# Create a function that creates a merged dataset using the gen_data and the sensor_data.
def create_gen_sensor_data(gen_data, sensor_data):
    # MERGED DATASET 2: CONTAINS GEN_DATA, SENSOR_DATA
    # Find the common IDs across the 3 datasets
    common_IDs2 = np.intersect1d(gen_data['ID'].values, sensor_data['ID'].values)
    # Only sensor data
    merged_data2 = gen_data[gen_data['ID'].isin(common_IDs2)]

    # Add columns from the sensor and lab tests to the respective merged datasets
    merged_data2 = create_sensor_cols(merged_data2)

    # Simply get the sensor data for the earliest test dates.
    sensor_data2_2 = earliest_dates('ID', 'Constant record date', sensor_data.copy(), gen_data.copy())

    # Fix the sensor data because they contain duplicate rows for every ID both for the copies of merged_data2
    sensor_data2_2 = fix_sensor_values(sensor_data2_2.copy())

    # Pass the sensor values in both copies of the merged_data and in merged_data2
    merged_data2 = pass_sensor_values(sensor_data2_2.copy(), merged_data2.copy())

    # Return the dataset
    return merged_data2


# Create a function that places the sensor tests as columns in the merged2 dataset and initializes them with NaN values.
def create_sensor_cols(merged_data2):
    # Add the new column names from sensor_data2 and place NaN values. For this part, I should at some point search for
    # the measurement units of these tests because they are not given in sensor_data2.
    sensor_col_names = ['Maximum blood pressure value', 'Minimum blood pressure value',
                     'Temperature value', 'Heart rate value', 'Oxygen saturation value', 'Blood glucose value']

    merged_data2.loc[:, sensor_col_names] = float('NaN')
    # Return the merged_data
    return merged_data2


# This function removes the columns of a data frame that have missing values for at least threshold % of their
# instances.
def remove_empty_columns(df, threshold):
    instances = df.shape[0]
    for x in df.columns:
        col = df[x].copy()
        unfilled_col = col[col.isna()]
        unfilled_instances = unfilled_col.shape[0]
        ratio = unfilled_instances/instances
        if ratio >= threshold:
            df.drop(x, axis=1, inplace=True)
        else:
            continue

    return df


# This function removes columns that contain the same values for at least threshold % of their instances.
def remove_undiluted_columns(df, threshold, accepted_cols):
    instances = df.shape[0]
    for x in df.columns:
        col = df[x].copy()
        values = col.unique()
        for i in values:
            common_values = col[col == i]
            com_num = len(common_values)
            ratio = com_num/instances
            if ratio >= threshold and x not in accepted_cols:
                df.drop(x, axis=1, inplace=True)
                break
            else:
                continue
    return df


# Create a function that removes the outliers from specific columns of the merged data and replaces their values
# with nan in order to use an imputation method afterwards to replace the nan values.
def replace_outliers_with_nan(merged_data, columns):
    # For every column to remove the outliers, find the .25 and .75 quantiles as well as the IQR and keep only those
    # values that are in 1.5*IQR of Q1 and Q3.
    if len(columns) == 1:
        data_to_clean = merged_data.loc[:, columns].copy().to_frame()
    else:
        data_to_clean = merged_data.loc[:, columns].copy()
    # Find the quantile params
    Q1 = data_to_clean.quantile(q=0.25)
    Q3 = data_to_clean.quantile(q=0.75)
    IQR = Q3 - Q1

    # Replace the outliers with nan values for every column
    for i in range(0, len(columns)):
        temp_col = data_to_clean[columns[i]].copy().to_frame()
        temp_col[((temp_col < (Q1[i] - 1.5*IQR[i])) | (temp_col > (Q3[i] + 1.5*IQR[i]))).any(axis=1)] = np.nan

        # Replace the column
        data_to_clean.loc[:, columns[i]] = temp_col.copy()

    # Return the cleaned data
    return data_to_clean


# Create a function that fills missing values in the dataset using multivariate feature imputation, where essentially
# a column with missing values is considered as the output and the rest columns are considered as an input to a linear
# regression model that predicts the missing values.
def fill_missing_values(df):
    # Initialize imputer model
    imp = IterativeImputer(max_iter=40, random_state=42)
    # Fit the dataset
    imp.fit(df)
    # Get the imputed dataset
    imputed_df = imp.transform(df)
    # Convert it back to a dataframe
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns, index=df.index)
    # Since the IterativeImputer converts all values to floats, convert the elements of columns with integer values back
    # to integer values
    int_columns = ['Age', 'Sex', 'Reason for discharge as Inpatient']
    imputed_df[int_columns] = imputed_df[int_columns].astype(int)
    # Return the imputed dataframe
    return imputed_df


# Create a function that receives the comorbidity dataset and merges certain diseases into 1 general category for easier
# use and drops those that are not relevant
def general_comorbidity_dataset(comorbidity_dataset):
    # Drop specific comorbidities that are irrelevant
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
    return


# Create a function that receives the comorbidity dataset and the 3 merged datasets and merges the comorbidity dataset
# with them for their common IDs
def create_merged_with_comorbidities(merged_data_no_td, merged_data_td, merged_data2, comorbidity_dataset):
    # Get the IDs of each dataset
    ids_no_td = merged_data_no_td['ID'].values
    ids_td = merged_data_td['ID'].values
    ids2 = merged_data2['ID'].values
    ids_com = comorbidity_dataset['ID'].values

    # Create 3 copies of the comorbidity_dataset, one for each merged dataset
    comorbidity_dataset_no_td = comorbidity_dataset[np.isin(comorbidity_dataset['ID'], ids_no_td)].copy()
    comorbidity_dataset_no_td = comorbidity_dataset_no_td.sort_values(by='ID', ascending=True)

    comorbidity_dataset_td = comorbidity_dataset[np.isin(comorbidity_dataset['ID'], ids_td)].copy()
    comorbidity_dataset_td = comorbidity_dataset_td.sort_values(by='ID', ascending=True)

    comorbidity_dataset2 = comorbidity_dataset[np.isin(comorbidity_dataset['ID'], ids2)].copy()
    comorbidity_dataset2 = comorbidity_dataset2.sort_values(by='ID', ascending=True)

    # Keep only the rows of the merged data that correspond to the comorbidity dataset ids
    merged_data_no_td_com = merged_data_no_td[np.isin(merged_data_no_td['ID'], ids_com)].copy()
    merged_data_td_com = merged_data_td[np.isin(merged_data_td['ID'], ids_com)].copy()
    merged_data2_com = merged_data2[np.isin(merged_data2['ID'], ids_com)].copy()

    # Make the comorbidity indices the same as the merged datasets
    comorbidity_dataset_no_td.index = merged_data_no_td_com.index.values
    comorbidity_dataset_td.index = merged_data_td_com.index.values
    comorbidity_dataset2.index = merged_data2_com.index.values

    # Merge the datasets without duplicating the ID column
    merged_data_no_td_com = merged_data_no_td_com.merge(comorbidity_dataset_no_td, left_on='ID', right_on='ID',
                                                        suffixes=('_left', '_right'))
    merged_data_td_com = merged_data_td_com.merge(comorbidity_dataset_td, left_on='ID', right_on='ID',
                                                  suffixes=('_left', '_right'))
    merged_data2_com = merged_data2_com.merge(comorbidity_dataset2, left_on='ID', right_on='ID',
                                              suffixes=('_left', '_right'))

    # Return the merged datasets
    return merged_data_no_td_com, merged_data_td_com, merged_data2_com


# Create a function that receives the ICD10 dataset, creates a dictionary for every ICD10 code that was found manually
# to be in the top 20 most frequent diseases of every diagnostic column (DIA_PPAL, DIA_02, etc) and decodes the ICD10
# code to disease name. For the diseases found at the top 20, view the google doc below
# https://docs.google.com/document/d/1lTjgRdHLvhnjK7dg65yd9RVsyz9Et7Drwb8qZLcsBQc/edit
# Afterwards, create a dataset with the columns: 'ID', 'Disease1', 'Disease2', etc and mark the specific cells with 1
# or 0 whether that ID has the specific disease or not. In order to place the 1 or 0 view the ICD10 dataset columns
# that contain that information, which are the columns 'POAD_PPAL', 'POAD_02', etc. S and E mean 1 and N means 0.
def create_comorbidity_dataset2(ICD10_data):
    # Get the IDs of the ICD10 dataset
    ids = ICD10_data['ID'].values
    # Initialize the comorbidity dataset
    comorbidity_dataset = pd.DataFrame(columns=['ID', 'Cardiac dysrhythmias', 'Chronic Kidney Disease',
                                                'Coronary atherosclerosis', 'Diabetes'], index=np.arange(0, len(ids)))

    comorbidity_dataset['ID'] = ids
    comorbidity_dataset.iloc[:, 1:] = 0

    dia_list = ['DIA_PPAL']
    poad_list = ['POAD_PPAL']
    temp1 = 'DIA_'
    temp2 = 'POAD_'
    for j in range(2, 20):
        if j <= 9:
            dia_list.append(temp1 + '0' + str(j))
            poad_list.append(temp2 + '0' + str(j))
        else:
            dia_list.append(temp1 + str(j))
            poad_list.append(temp2 + str(j))

    all_values = list()
    for i in range(0, len(dia_list)):
        all_values = all_values + ICD10_data.loc[:, dia_list[i]].values.tolist()

    all_values = np.unique(all_values)
    diabetes_list = list()
    heart_list = list()
    kidney_list = list()
    atherosclerosis_list = list()
    for i in range(0, len(all_values)):
        value = all_values[i]
        if 'E' in value:
            diabetes_list.append(value)
        elif 'I' in value:
            heart_list.append(value)
        elif 'N' in value:
            kidney_list.append(value)

    # Remove unwanted codes
    heart_list.remove('I82.4Z1')
    heart_list.remove('I82.4Z2')
    heart_list.remove('I82.B11')
    heart_list.remove('I82.B12')
    heart_list.remove('I82.C11')
    heart_list.remove('I82.C12')
    heart_list.remove('I82.C19')

    # Remove the letter
    diabetes_list = [float(i.split('E')[1]) for i in diabetes_list]
    heart_list = [float(i.split('I')[1]) for i in heart_list]
    kidney_list = [float(i.split('N')[1]) for i in kidney_list]

    # Keep only the ones we need
    # Diabetes codes in E08-E13
    diabetes_list = np.array(diabetes_list)
    diabetes_list = diabetes_list[diabetes_list >= 8]
    diabetes_list = diabetes_list[diabetes_list < 14]

    # Coronary atherosclerosis and other heart disease in I25
    heart_list = np.array(heart_list)
    atherosclerosis_list = heart_list[heart_list >= 25]
    atherosclerosis_list = heart_list[heart_list < 26]

    # Cardiac arrhythmias in I49.0 - I49.9
    heart_list = heart_list[heart_list >= 49]
    heart_list = heart_list[heart_list < 50]

    # Chronic Kidney Disease in N18.0 - N18.9
    kidney_list = np.array(kidney_list)
    kidney_list = kidney_list[kidney_list >= 18]
    kidney_list = kidney_list[kidney_list < 19]

    # Convert the numbers back to strings with the codes
    diabetes_list = ['E' + str(i) for i in diabetes_list]
    atherosclerosis_list = ['I' + str(i) for i in atherosclerosis_list]
    heart_list = ['I' + str(i) for i in heart_list]
    kidney_list = ['N' + str(i) for i in kidney_list]

    # For every ID get its row, find the POAD columns for which it has 'S' or 'E' and then go to the respective DIA
    # column and use the disease dictionary to mark the corresponding cell in the comorbidity_dataset.
    for i in range(0, len(ids)):
        # Get the id
        id = ids[i]
        # Get the row
        row = ICD10_data[ICD10_data['ID'] == id]
        # Get the POAD columns that have S or E values
        poad_columns = row[poad_list]
        poad_columns = poad_columns[(poad_columns == 'S') | (poad_columns == 'E')]
        poad_columns = poad_columns.columns.values
        # Keep the index part (after the _ ) of the column name to use for the DIA columns
        poad_index = [j.split('_')[1] for j in poad_columns]
        # Now find the ICD10 disease keys of the row
        row_keys = row.loc[:, ['DIA_' + j for j in poad_index]].values.tolist()[0]

        # Get the index value of the id in the comorbidity dataset
        id_index = comorbidity_dataset[comorbidity_dataset['ID'] == id].index.values
        # Check whether the available columns contain a general disease category from the 4 above
        # Diabetes check
        if len(np.intersect1d(diabetes_list, row_keys)) >= 1:
            comorbidity_dataset.loc[id_index, 'Diabetes'] = 1
        #
        if len(np.intersect1d(heart_list, row_keys)) >= 1:
            comorbidity_dataset.loc[id_index, 'Cardiac dysrhythmias'] = 1
        #
        if len(np.intersect1d(atherosclerosis_list, row_keys)) >= 1:
            comorbidity_dataset.loc[id_index, 'Coronary atherosclerosis'] = 1
        #
        if len(np.intersect1d(kidney_list, row_keys)) >= 1:
            comorbidity_dataset.loc[id_index, 'Chronic Kidney Disease'] = 1

    # Return the comorbidity dataset
    return comorbidity_dataset


# Create a function that receives the X_train, y_train, X_val, y_val, X_test and y_test sets and fills their missing
# values using the filled values of the corresponding set.
def fill_train_valid_test_sets(X_train, y_train, X_val, y_val, X_test, y_test):
    # TRAIN
    X_train['Reason for discharge as Inpatient'] = y_train
    X_train = fill_missing_values(X_train)
    X_train = X_train.drop('Reason for discharge as Inpatient', axis=1)

    # VAL
    X_val['Reason for discharge as Inpatient'] = y_val
    X_val = fill_missing_values(X_val)
    X_val = X_val.drop('Reason for discharge as Inpatient', axis=1)

    # TEST
    X_test['Reason for discharge as Inpatient'] = y_test
    X_test = fill_missing_values(X_test)
    X_test = X_test.drop('Reason for discharge as Inpatient', axis=1)

    # Returned the features of the filled sets. The classes remain the same.
    return X_train, X_val, X_test


# Create a function that keeps the rows that have at least missing_threshold * 100% of their columns filled
def drop_rows_with_missing_values(df, missing_threshold):
    # Get features
    features = df.columns.values

    # Round down the number of columns that must be filled based on the threshold
    filled_col_num = np.floor(missing_threshold*len(features))

    # Initialize the row indices to drop as well as the number of nan values per row
    rows_to_drop = list()
    row_nan_num = list()

    # For every row get the number of missing values
    for i in df.index.values:
        row_nan = df.loc[[i]].isna().sum().sum()
        row_nan_num.append(row_nan)
        if len(features) - row_nan < filled_col_num:
            rows_to_drop.append(i)

    # Keep only the rows that pass the missing_threshold
    accepted_rows = np.setdiff1d(df.index.values, rows_to_drop)
    df = df.loc[accepted_rows, :]

    return df, rows_to_drop, row_nan_num


# Create a function that receives a list of models and filters them based on specific performance metric thresholds:
# 1) precision_th: The minimum precision value that the models must have in order to be plotted (default = 0.8)
# 2) mcc_th: The minimum mcc value that the models must have in order to be plotted (default = 0.5)
# 3) rec_th: The minimum recall value that the models must have in order to be plotted (default = 0.4)
# 4) ap_th: The minimum average precision value that the models must have in order to be plotted (default = 0.6)
# 5) auc_th: The minimum auc value that the models must have in order to be plotted (default = 0.85)
# 6) f1_th: The minimum f1-score value that the models must have in order to be plotted (default = 0.5)
def filter_models_based_on_performance_metrics(performance_dataframe_val, performance_dataframe_test, eval_set,
                                               precision_th=0.8, mcc_th=0.5, rec_th=0.4, ap_th=0.6, auc_th=0.85,
                                               f1_th=0.5):
    
    if eval_set == 'Val':
        # Keep only the models that satisfy the performance metric thresholds
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - Precision'] >= precision_th]
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - MCC'] >= mcc_th]
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - Recall'] >= rec_th]
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - Average Precision'] >= ap_th]
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - AUC'] >= auc_th]
        performance_dataframe_val = performance_dataframe_val[performance_dataframe_val[eval_set + ' - F-score'] >= f1_th]

        # Get indices of best models and keep the same for the other eval set
        model_indices = performance_dataframe_val.index.values
        performance_dataframe_test = performance_dataframe_test.loc[model_indices, :]
        
    elif eval_set == 'Test':
        # Keep only the models that satisfy the performance metric thresholds
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - Precision'] >= precision_th]
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - MCC'] >= mcc_th]
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - Recall'] >= rec_th]
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - Average Precision'] >= ap_th]
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - AUC'] >= auc_th]
        performance_dataframe_test = performance_dataframe_test[performance_dataframe_test[eval_set + ' - F-score'] >= f1_th]

        # Get indices of best models and keep the same for the other eval set
        model_indices = performance_dataframe_test.index.values
        performance_dataframe_val = performance_dataframe_val.loc[model_indices, :]

    # Return the filtered model dataframes
    return performance_dataframe_val, performance_dataframe_test


# Create a function that receives a list of models and keeps the ones that perform relatively the same both in the val
# and the test set using a z-score threshold
def keep_balanced_models(performance_dataframe_val, performance_dataframe_test, z_score):
    # Find the difference between the val and the test performances on the same model
    model_dif = pd.DataFrame()
    model_dif['Precision'] = performance_dataframe_val.loc[:, 'Val - Precision'] - performance_dataframe_test.loc[:, 'Test - Precision']
    model_dif['MCC'] = performance_dataframe_val.loc[:, 'Val - MCC'] - performance_dataframe_test.loc[:, 'Test - MCC']
    model_dif['Recall'] = performance_dataframe_val.loc[:, 'Val - Recall'] - performance_dataframe_test.loc[:, 'Test - Recall']
    model_dif['Average Precision'] = performance_dataframe_val.loc[:, 'Val - Average Precision'] - performance_dataframe_test.loc[:, 'Test - Average Precision']
    model_dif['AUC'] = performance_dataframe_val.loc[:, 'Val - AUC'] - performance_dataframe_test.loc[:, 'Test - AUC']
    model_dif['F-score'] = performance_dataframe_val.loc[:, 'Val - F-score'] - performance_dataframe_test.loc[:, 'Test - F-score']
    model_dif['Accuracy'] = performance_dataframe_val.loc[:, 'Val - Accuracy'] - performance_dataframe_test.loc[:, 'Test - Accuracy']

    # Get the z_score values for all models based on the model difs
    z_score_dif = pd.DataFrame()
    z_score_dif['Precision'] = (model_dif.loc[:, 'Precision'] - model_dif.loc[:, 'Precision'].mean())/model_dif.loc[:, 'Precision'].std()
    z_score_dif['MCC'] = (model_dif.loc[:, 'MCC'] - model_dif.loc[:, 'MCC'].mean()) / model_dif.loc[:, 'MCC'].std()
    z_score_dif['Recall'] = (model_dif.loc[:, 'Recall'] - model_dif.loc[:, 'Recall'].mean()) / model_dif.loc[:, 'Recall'].std()
    z_score_dif['Average Precision'] = (model_dif.loc[:, 'Average Precision'] - model_dif.loc[:, 'Average Precision'].mean()) / model_dif.loc[:, 'Average Precision'].std()
    z_score_dif['AUC'] = (model_dif.loc[:, 'AUC'] - model_dif.loc[:, 'AUC'].mean()) / model_dif.loc[:, 'AUC'].std()
    z_score_dif['F-score'] = (model_dif.loc[:, 'F-score'] - model_dif.loc[:, 'F-score'].mean()) / model_dif.loc[:, 'F-score'].std()
    z_score_dif['Accuracy'] = (model_dif.loc[:, 'Accuracy'] - model_dif.loc[:, 'F-score'].mean()) / model_dif.loc[:, 'Accuracy'].std()

    # Keep models that are above a certain z-score threshold for every metric
    models_to_keep = z_score_dif.copy()
    models_to_keep = models_to_keep[models_to_keep['Precision'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['MCC'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['Recall'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['Average Precision'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['AUC'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['F-score'] >= z_score]
    models_to_keep = models_to_keep[models_to_keep['Accuracy'] >= z_score]

    # Return results
    return model_dif, z_score_dif, models_to_keep