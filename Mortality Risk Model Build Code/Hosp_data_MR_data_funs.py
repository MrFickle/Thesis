"""
This script contains functions that are used only for loading or storing data.
"""

# Modules
import pandas as pd
import numpy as np
import Hosp_data_MR_proc_funs as procfun


# Create a function that loads the translated datasets required
def load_translated_covid_data():
    # 1) COVID_DSL_01 - GENERAL DATA
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data' + \
                r'\gen_data_COVID_DSL_01_DATA_ENG.csv'
    gen_data = pd.read_csv(data_path, low_memory=False, encoding='latin-1', index_col=0)

    # 2) COVID_DSL_02 - SENSOR DATA
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data' + \
                r'\sensor_data_COVID_DSL_02_DATA_ENG.csv'
    sensor_data = pd.read_csv(data_path, low_memory=False, encoding='latin-1', index_col=0)

    # 3) COVID_DSL_06 - LAB DATA
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data' + \
                r'\lab_data_COVID_DSL_06_DATA_ENG.csv'
    lab_data = pd.read_csv(data_path, low_memory=False, encoding='latin-1', index_col=0)

    # Return the data
    return gen_data, sensor_data, lab_data


# Create a function that loads the ICD10 datasets from 19/04/2021 (latest data as of 08/07/2021) with the
# right encoding and stores them translated.
def store_ICD10_translated_covid_data():
    # 1) COVID_DSL_03
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\COVID_DSL_03.CSV'
    COVID_DSL_03_DATA = pd.read_csv(data_path, low_memory=False, sep='|', encoding='latin-1')

    # Translate columns
    COVID_DSL_03_DATA.rename(columns={'IDINGRESO': 'ID'}, inplace=True)

    # Sort the data using the ID values
    COVID_DSL_03_DATA.sort_values(by='ID', inplace=True)

    # Store the translated general data
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data' + \
                r'\ICD10_data_COVID_DSL_03_DATA_ENG.csv'

    COVID_DSL_03_DATA.to_csv(data_path, index=True, header=True)


    # 2) COVID_DSL_05
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\COVID_DSL_05.CSV'
    COVID_DSL_05_DATA = pd.read_csv(data_path, low_memory=False, sep='|', encoding='latin-1')

    # Translate columns
    COVID_DSL_05_DATA.rename(columns={'IDINGRESO': 'ID'}, inplace=True)

    # Swap the first column DIA_PPAL with the ID column
    cols = list(COVID_DSL_05_DATA.columns)
    a, b = cols.index('DIA_PPAL'), cols.index('ID')
    cols[b], cols[a] = cols[a], cols[b]
    COVID_DSL_05_DATA = COVID_DSL_05_DATA[cols]

    # Sort the data using the ID values
    COVID_DSL_05_DATA.sort_values(by='ID', inplace=True)

    # Store the translated general data
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data' + \
                r'\ICD10_data_COVID_DSL_05_DATA_ENG.csv'

    COVID_DSL_05_DATA.to_csv(data_path, index=True, header=True)


# Create a function that loads the translated ICD10 datasets from 19/04/2021 (latest data as of 08/07/2021) with the
# right encoding.
def load_ICD10_translated_covid_data():
    # COVID_DSL_05
    filename = r'\ICD10_data_COVID_DSL_05_DATA_ENG.csv'

    # Data path
    data_path = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\19_04_2021\Translated Data'

    # Load the data
    ICD10_data = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    # Return the data
    return ICD10_data


# Create a function that stores the merged data before we place the comorbidities
def store_merged_before_comorbidities(merged_data_no_td, merged_data_td, merged_data2):
    # Folder name
    folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
             r'\Before merging comorbidities'

    # Filenames
    filename1 = r'\merged_data_no_td.csv'
    filename2 = r'\merged_data_td.csv'
    filename3 = r'\merged_data2.csv'

    # Store the files
    merged_data_no_td.to_csv(folder + filename1, index=True, header=True)
    merged_data_td.to_csv(folder + filename2, index=True, header=True)
    merged_data2.to_csv(folder + filename3, index=True, header=True)


# Create a function that loads the merged data before we place the comorbidities
def load_merged_before_comorbidities():
    # Folder name
    folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
             r'\Before merging comorbidities'

    # Filenames
    filename1 = r'\merged_data_no_td.csv'
    filename2 = r'\merged_data_td.csv'
    filename3 = r'\merged_data2.csv'

    # Load the files
    merged_data_no_td = pd.read_csv(folder + filename1, index_col=0, low_memory=False)
    merged_data_td = pd.read_csv(folder + filename2, index_col=0, low_memory=False)
    merged_data2 = pd.read_csv(folder + filename3, index_col=0, low_memory=False)

    # Return the data
    return merged_data_no_td, merged_data_td, merged_data2


# Create a function that stores the merged data after we place the comorbidities
def store_merged_after_comorbidities(merged_data_no_td_com, merged_data_td_com, merged_data2_com, imp_case):
    if imp_case == 1:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
                 r'\After merging comorbidities'
    elif imp_case == 2:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
                 r'\Covidanalytics format'

    # Filenames
    filename1 = r'\merged_data_no_td_com.csv'
    filename2 = r'\merged_data_td_com.csv'
    filename3 = r'\merged_data2_com.csv'

    # Store the files
    merged_data_no_td_com.to_csv(folder + filename1, index=True, header=True)
    merged_data_td_com.to_csv(folder + filename2, index=True, header=True)
    merged_data2_com.to_csv(folder + filename3, index=True, header=True)


# Create a function that loads the merged data before we place the comorbidities
def load_merged_after_comorbidities(imp_case):
    if imp_case == 1:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
                 r'\After merging comorbidities'
    elif imp_case == 2:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Mortality Risk' + \
                 r'\Covidanalytics format'

    # Filenames
    filename1 = r'\merged_data_no_td_com.csv'
    filename2 = r'\merged_data_td_com.csv'
    filename3 = r'\merged_data2_com.csv'

    # Load the files
    merged_data_no_td_com = pd.read_csv(folder + filename1, index_col=0, low_memory=False)
    merged_data_td_com = pd.read_csv(folder + filename2, index_col=0, low_memory=False)
    merged_data2_com = pd.read_csv(folder + filename3, index_col=0, low_memory=False)

    # Return the data
    return merged_data_no_td_com, merged_data_td_com, merged_data2_com


# Create a function that loads the merged data before we place the comorbidities
def load_merged_SR_after_comorbidities(imp_case):
    if imp_case == 1:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Severity Risk' + \
                 r'\After merging comorbidities'
    elif imp_case == 2:
        # Folder name
        folder = r'H:\Documents\Coding\Projects\Thesis\Dataset\Hosp Data\After processing\19_04_2021\Severity Risk' + \
                 r'\Covidanalytics format'

    # Filenames
    filename1 = r'\merged_data_no_td_com.csv'
    filename2 = r'\merged_data_td_com.csv'
    filename3 = r'\merged_data2_com.csv'

    # Load the files
    merged_data_no_td_com = pd.read_csv(folder + filename1, index_col=0, low_memory=False)
    merged_data_td_com = pd.read_csv(folder + filename2, index_col=0, low_memory=False)
    merged_data2_com = pd.read_csv(folder + filename3, index_col=0, low_memory=False)

    # Return the data
    return merged_data_no_td_com, merged_data_td_com, merged_data2_com


# Create a function that loads the desired merged data after the comorbidities, while defining the type of data to use,
# creating 4 general categories:
# 1) All patients with all comorbidities and severity
# 2) All patients with all comorbidities
# 3) Patients with a specific comorbidity or with severity separately from the patients without that specific
# comorbidity or severity (contains severity like (1) )
# 4) Patients with a specific comorbidity separately from the patients without that specific comorbidity (does not
# contain severity like (2) )
def load_specific_data(data_name, category, sub_category, th):
    # Category 1
    if category == 1:
        merged_data_no_td, merged_data_td, merged_data2 = load_merged_after_comorbidities(2)
        merged_data_no_td_sr, merged_data_td_sr, merged_data2_sr = load_merged_SR_after_comorbidities(2)
        if data_name == 'merged data no td':
            IDs = merged_data_no_td['ID'].values
            IDs_sr = merged_data_no_td_sr['ID'].values
            common_ids = np.intersect1d(IDs, IDs_sr)
            merged_data_no_td_sr = merged_data_no_td_sr[merged_data_no_td_sr['ID'].isin(common_ids)]
            merged_data_no_td = merged_data_no_td[merged_data_no_td['ID'].isin(common_ids)]
            merged_data_no_td_sr = merged_data_no_td_sr.sort_values(by='ID')
            merged_data_no_td = merged_data_no_td.sort_values(by='ID')
            merged_data_no_td['Severity'] = 0
            merged_data_no_td['Severity'] = merged_data_no_td_sr['Severity'].values
            data = merged_data_no_td.copy()
        elif data_name == 'merged data2':
            IDs = merged_data2['ID'].values
            IDs_sr = merged_data2_sr['ID'].values
            common_ids = np.intersect1d(IDs, IDs_sr)
            merged_data2_sr = merged_data2_sr[merged_data2_sr['ID'].isin(common_ids)]
            merged_data2 = merged_data2[merged_data2['ID'].isin(common_ids)]
            merged_data2_sr = merged_data2_sr.sort_values(by='ID')
            merged_data2 = merged_data2.sort_values(by='ID')
            merged_data2['Severity'] = 0
            merged_data2['Severity'] = merged_data2_sr['Severity'].values
            data = merged_data2.copy()

        # Drop unwanted columns
        data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
        # Replace certain unwanted string characters
        data.columns = data.columns.str.replace(',', '')
        data.columns = data.columns.str.replace('[', '(')
        data.columns = data.columns.str.replace(']', ')')

        # Keep only adult people
        data = data[data['Age'] >= 18]

        # Keep only patients that have at least th% of their values filled
        data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

    # Category 2
    elif category == 2:
        merged_data_no_td, merged_data_td, merged_data2 = load_merged_after_comorbidities(2)
        if data_name == 'merged data no td':
            data = merged_data_no_td.copy()
        elif data_name == 'merged data2':
            data = merged_data2.copy()

        # Drop unwanted columns
        data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
        # Replace certain unwanted string characters
        data.columns = data.columns.str.replace(',', '')
        data.columns = data.columns.str.replace('[', '(')
        data.columns = data.columns.str.replace(']', ')')

        # Keep only adult people
        data = data[data['Age'] >= 18]

        # Keep only patients that have at least th% of their values filled
        data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

    # Category 3
    elif category == 3:
        merged_data_no_td, merged_data_td, merged_data2 = load_merged_after_comorbidities(2)
        merged_data_no_td_sr, merged_data_td_sr, merged_data2_sr = load_merged_SR_after_comorbidities(2)
        if data_name == 'merged data no td':
            IDs = merged_data_no_td['ID'].values
            IDs_sr = merged_data_no_td_sr['ID'].values
            common_ids = np.intersect1d(IDs, IDs_sr)
            merged_data_no_td_sr = merged_data_no_td_sr[merged_data_no_td_sr['ID'].isin(common_ids)]
            merged_data_no_td = merged_data_no_td[merged_data_no_td['ID'].isin(common_ids)]
            merged_data_no_td_sr = merged_data_no_td_sr.sort_values(by='ID')
            merged_data_no_td = merged_data_no_td.sort_values(by='ID')
            merged_data_no_td['Severity'] = 0
            merged_data_no_td['Severity'] = merged_data_no_td_sr['Severity'].values
            data = merged_data_no_td.copy()


        elif data_name == 'merged data2':
            IDs = merged_data2['ID'].values
            IDs_sr = merged_data2_sr['ID'].values
            common_ids = np.intersect1d(IDs, IDs_sr)
            merged_data2_sr = merged_data2_sr[merged_data2_sr['ID'].isin(common_ids)]
            merged_data2 = merged_data2[merged_data2['ID'].isin(common_ids)]
            merged_data2_sr = merged_data2_sr.sort_values(by='ID')
            merged_data2 = merged_data2.sort_values(by='ID')
            merged_data2['Severity'] = 0
            merged_data2['Severity'] = merged_data2_sr['Severity'].values
            data = merged_data2.copy()

        # Drop unwanted columns
        data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
        # Replace certain unwanted string characters
        data.columns = data.columns.str.replace(',', '')
        data.columns = data.columns.str.replace('[', '(')
        data.columns = data.columns.str.replace(']', ')')

        # Keep only adult people
        data = data[data['Age'] >= 18]

        # Keep only patients that have at least th% of their values filled
        data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

        if sub_category in ['Severity', 'Coronary atherosclerosis', 'Diabetes']:
            data1 = data[data[sub_category] == 1]
            data2 = data[data[sub_category] == 0]
            data = [data1, data2]
        else:
            print('Wrong sub_category given. Accepted values are: Severity, Coronary atherosclerosis and Diabetes')

    # Category 4
    elif category == 4:
        merged_data_no_td, merged_data_td, merged_data2 = load_merged_after_comorbidities(2)
        if data_name == 'merged data no td':
            data = merged_data_no_td.copy()
        elif data_name == 'merged data2':
            data = merged_data2.copy()

        # Drop unwanted columns
        data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
        # Replace certain unwanted string characters
        data.columns = data.columns.str.replace(',', '')
        data.columns = data.columns.str.replace('[', '(')
        data.columns = data.columns.str.replace(']', ')')

        # Keep only adult people
        data = data[data['Age'] >= 18]

        # Keep only patients that have at least th% of their values filled
        data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

        if sub_category in ['Coronary atherosclerosis', 'Diabetes']:
            data1 = data[data[sub_category] == 1]
            data2 = data[data[sub_category] == 0]
            data = [data1, data2]
        else:
            print('Wrong sub_category given. Accepted values are: Coronary atherosclerosis and Diabetes')

    # Return the data
    return data


# Create a function that loads all the models formed from grid search
def load_gridsearch_models(folder_val, folder_test, th):
    # Define the names of the evaluation metrics used to train the models using randomized grid search, load the
    # models for one metric at a time and place them all together in a single dataframe
    scoring_names = ['error', 'logloss', 'auc', 'aucpr', 'map', 'f1-score', 'balanced accuracy', 'precision', 'recall',
                     'mcc']

    # Initialize dataframe for all evaluation metrics used on the validation set and on the test set for all
    # approaches
    performance_dataframe_val = pd.DataFrame()
    performance_dataframe_test = pd.DataFrame()

    # Load all models
    for i in range(0, len(scoring_names)):
        # Val set
        filename_val = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
            int(100 * th)) + '.csv'
        temp_data_val = pd.read_csv(folder_val + filename_val, index_col=0,
                                    low_memory=False)
        # Sort by the indices
        temp_data_val = temp_data_val.sort_index()
        # Append data
        performance_dataframe_val = performance_dataframe_val.append(temp_data_val)

        # Test set
        filename_test = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
            int(100 * th)) + '.csv'
        temp_data_test = pd.read_csv(folder_test + filename_test, index_col=0,
                                     low_memory=False)
        # Sort by the indices
        temp_data_test = temp_data_test.sort_index()
        # Append data
        performance_dataframe_test = performance_dataframe_test.append(temp_data_test)

    # Reset the indices
    performance_dataframe_val.index = np.arange(0, performance_dataframe_val.shape[0])
    performance_dataframe_test.index = np.arange(0, performance_dataframe_test.shape[0])

    # Drop rows that have identical performance metrics
    performance_metrics_cols_val = ['Val - Average Precision', 'Val - Precision', 'Val - Accuracy',
                                    'Val - Recall', 'Val - F-score', 'Val - MCC', 'Val - AUC']
    # performance_metrics_cols_test = ['Test - Average Precision', 'Test - Precision', 'Test - Accuracy',
    #                                 'Test - Recall', 'Test - F-score', 'Test - MCC', 'Test - AUC']
    #for i in performance_metrics_cols_val:
    #    performance_dataframe_val.loc[:, i] = round(performance_dataframe_val.loc[:, i], 2)
    # for i in performance_metrics_cols_test:
    #    performance_dataframe_test.loc[:, i] = round(performance_dataframe_test.loc[:, i], 2)

    performance_dataframe_val = performance_dataframe_val.drop_duplicates(subset=performance_metrics_cols_val,
                                                                          keep='first')
    performance_dataframe_test = performance_dataframe_test.loc[performance_dataframe_val.index.values, :]

    # Return models
    return performance_dataframe_val, performance_dataframe_test


# Store the performance metric values from the 9-fold cv performed on the filtered models as well as the models built
# from the grid search
def store_cv_models(mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val,
                    std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test,
                    performance_dataframe_val, performance_dataframe_test, data_path, th):
    filename = r'\mean_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    mean_performance_nfold_cv_val.to_csv(data_path + filename, index=True, header=True)

    filename = r'\mean_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    mean_performance_nfold_cv_test.to_csv(data_path + filename, index=True, header=True)

    filename = r'\std_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    std_performance_nfold_cv_val.to_csv(data_path + filename, index=True, header=True)

    filename = r'\std_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    std_performance_nfold_cv_test.to_csv(data_path + filename, index=True, header=True)

    filename = r'\sem_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    sem_performance_nfold_cv_val.to_csv(data_path + filename, index=True, header=True)

    filename = r'\sem_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    sem_performance_nfold_cv_test.to_csv(data_path + filename, index=True, header=True)

    filename = r'\performance_val_data' + str(int(100*th)) + '.csv'
    performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

    filename = r'\performance_test_data' + str(int(100*th)) + '.csv'
    performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)


# Load the performance metric values from the 9-fold cv performed on the filtered models as well as the models built
# from the grid search
def load_cv_models(data_path, th):
    filename = r'\mean_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    mean_performance_nfold_cv_val = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\mean_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    mean_performance_nfold_cv_test = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\std_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    std_performance_nfold_cv_val = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\std_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    std_performance_nfold_cv_test = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\sem_performance_nfold_cv_val_data' + str(int(100*th)) + '.csv'
    sem_performance_nfold_cv_val = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\sem_performance_nfold_cv_test_data' + str(int(100*th)) + '.csv'
    sem_performance_nfold_cv_test = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\performance_val_data' + str(int(100*th)) + '.csv'
    performance_dataframe_val = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    filename = r'\performance_test_data' + str(int(100*th)) + '.csv'
    performance_dataframe_test = pd.read_csv(data_path + filename, index_col=0, low_memory=False)

    # Return the data
    return mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val, \
           std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test, \
           performance_dataframe_val, performance_dataframe_test
