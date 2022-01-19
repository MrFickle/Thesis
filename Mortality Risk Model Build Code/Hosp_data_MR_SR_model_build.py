# Modules
import pandas as pd
import numpy as np
from importlib import reload
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun
import random
import time


'''
GENERAL STEPS
'''
start_time = time.time()
# 1) Load pre-processed datasets
# merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_before_comorbidities()
merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_after_comorbidities(2)
merged_data_no_td_sr, merged_data_td_sr, merged_data2_sr = datafun.load_merged_SR_after_comorbidities(2)
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
folder = r'H:\Documents\Coding\Projects\Thesis\Models\Test'

'''
MERGED DATA NO TD
'''
datasets = 0
th = 0.85
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

# 6) Initialize a dataframe that contains the evaluation metric used with and for the best model produced, show
# the evaluation metric value as well as the general machine learning performance metrics.
# eval_metrics = ['balanced_accuracy', 'average_precision', 'f1', 'neg_log_loss', 'roc_auc', 'precision']
# scoring_functions = [procfun.xgb_balanced_accuracy, 'map', procfun.xgb_f1, 'logloss', 'auc', procfun.xgb_precision]
# scoring_functions = ['error', 'logloss', 'auc', 'aucpr', 'map']
scoring_functions = ['error', 'logloss', 'auc', 'aucpr', 'map', modelfun.xgb_f1, modelfun.xgb_balanced_accuracy,
                     modelfun.xgb_precision, modelfun.xgb_recall, modelfun.xgb_mcc]
scoring_names = ['error', 'logloss', 'auc', 'aucpr', 'map', 'f1-score', 'balanced accuracy', 'precision',
                 'recall', 'mcc']

# Define boosting rounds and early stopping rounds
boost_rounds = 1000
stopping_rounds = 50

# Define random grid
random_grid_params = [
    (max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight)
    for max_depth in [5, 6, 7, 8, 9, 10]
    for min_child_weight in [1, 2, 3, 4, 5, 6]
    for eta in [0.05]
    for subsample in [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for colsample_bytree in [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for scale_pos_weight in [1, 2, 3, 4]
]

# Select 20% of the pairs at random
gridsearch_params = random.sample(random_grid_params, round(len(random_grid_params)/5))

# Choose approach and define grids to use
# APPROACH 1
approach = 1
smote_params = []
modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                        smote_params, boost_rounds, stopping_rounds, folder, th)

print('Total time elapsed :', round(time.time()) - start_time)
