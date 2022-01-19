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
# 1) Load specific datasets
data_name = 'merged data no td'
category = 4
sub_category = 'Diabetes'
th = 0.85

if category in [3, 4]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)
    data1 = data[0]
    data2 = data[1]
elif category in [1, 2]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)

# 2) Define the scoring functions to use
scoring_functions = ['error', 'logloss', 'auc', 'aucpr', 'map', modelfun.xgb_f1, modelfun.xgb_balanced_accuracy,
                     modelfun.xgb_precision, modelfun.xgb_recall, modelfun.xgb_mcc]
scoring_names = ['error', 'logloss', 'auc', 'aucpr', 'map', 'f1-score', 'balanced accuracy', 'precision',
                 'recall', 'mcc']

# 3) Define boosting rounds and early stopping rounds
boost_rounds = 1000
stopping_rounds = 50

# 4) Define random grid
random_grid_params = [
    (max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight)
    for max_depth in [5, 6, 7, 8, 9, 10]
    for min_child_weight in [1, 2, 3, 4, 5, 6]
    for eta in [0.05]
    for subsample in [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for colsample_bytree in [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for scale_pos_weight in [1, 2, 3, 4]
]

# 5) Select 10% of the pairs at random
gridsearch_params = random.sample(random_grid_params, round(len(random_grid_params)/10))

# 6) Define folder to store results and choose approach
folder = r'H:\Documents\Coding\Projects\Thesis\Models\Test'
approach = 1
smote_params = []

# 7) Build models and get results
if approach == 1:
    modelfun.choose_grid_search_ML_approach(data2, approach, scoring_functions, scoring_names, gridsearch_params,
                                            smote_params, boost_rounds, stopping_rounds, folder, th)
elif approach == 7:
    modelfun.build_xgboost_model_group(data, approach, scoring_functions, scoring_names, gridsearch_params, smote_params,
                              boost_rounds, stopping_rounds, folder, th, 0.15)

print('Total time elapsed :', round(time.time()) - start_time)

