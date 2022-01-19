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

'''
MERGED DATA NO TD
'''
for datasets in [0, 1]:
    # 2) Choose one of the above datasets to train and drop unwanted columns.
    if datasets == 0:
        data = merged_data_no_td.copy()
        folder = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data no td com\Tuning 3'

        # Use missing value thresholds: 0.8, 0.85, 0.9
        missing_values_th = [0.8, 0.85, 0.9]

    elif datasets == 1:
        data = merged_data2.copy()
        folder = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data2 com\Tuning 3'

        # Use missing value thresholds: 0.9, 0.99
        missing_values_th = [0.9, 0.99]

    # Drop unwanted columns
    data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
    # Replace certain unwanted string characters
    data.columns = data.columns.str.replace(',', '')
    data.columns = data.columns.str.replace('[', '(')
    data.columns = data.columns.str.replace(']', ')')

    # Keep only adult people
    data = data[data['Age'] >= 18]

    for th in missing_values_th:
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
        stopping_rounds = 20

        # Define random grid
        random_grid_params = [
            (max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight)
            for max_depth in [6, 7, 8, 9, 10]
            for min_child_weight in [1, 2, 3, 4, 5]
            for eta in [0.05, 0.1]
            for subsample in [0.6, 0.8, 1]
            for colsample_bytree in [0.6, 0.8, 1]
            for scale_pos_weight in [1, 1.5, 2, 2.5]
        ]

        # Select 10% of the pairs at random
        gridsearch_params = random.sample(random_grid_params, round(len(random_grid_params)/10))

        # Choose approach and define grids to use
        # APPROACH 1
        approach = 1
        smote_params = []
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                smote_params, boost_rounds, stopping_rounds, folder, th)

        # APPROACH 2
        approach = 2
        smote_params = []
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                smote_params, boost_rounds, stopping_rounds, folder, th)

        # APPROACH 3
        approach = 3
        class_ratios = [0.35, 0.4, 0.45, 0.5]
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                class_ratios, boost_rounds, stopping_rounds, folder, th)

        # APPROACH 4
        approach = 4
        class_ratios = [0.35, 0.4, 0.45, 0.5]
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                class_ratios, boost_rounds, stopping_rounds, folder, th)


        # APPROACH 5
        approach = 5
        smote_grid = [
            (smote_ratio, smote_neighbours)
            for smote_ratio in [0.2, 0.25, 0.3]
            for smote_neighbours in [5, 8]
        ]
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                smote_grid, boost_rounds, stopping_rounds, folder, th)

        # APPROACH 6
        approach = 6
        smote_grid = [
            (smote_ratio, smote_neighbours, class_ratio)
            for smote_ratio in [0.2, 0.25, 0.3]
            for smote_neighbours in [5, 8]
            for class_ratio in [0.35, 0.4, 0.45]
        ]
        modelfun.choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params,
                                                smote_grid, boost_rounds, stopping_rounds, folder, th)

print('Total time elapsed :', round(time.time()) - start_time)