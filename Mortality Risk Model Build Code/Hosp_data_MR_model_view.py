# Modules
import pandas as pd
import numpy as np
from importlib import reload
import xgboost as xgb
from sklearn.model_selection import train_test_split
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun


'''
GENERAL STEPS
'''
# 1) Load pre-processed datasets
# merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_before_comorbidities()
merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_after_comorbidities(2)
merged_data_no_td_sr, merged_data_td_sr, merged_data2_sr = datafun.load_merged_SR_after_comorbidities(2)
IDs = merged_data_no_td['ID'].values
merged_data_td['Severity'] = 0
for i in IDs:
    row = merged_data_no_td[merged_data_no_td]

'''
MERGED DATA NO TD
'''
for datasets in [0, 1]:
# 2) Choose one of the above datasets to train and drop unwanted columns.
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

        # 3) Split into independent (X) and dependent variables (y)
        X = data.drop(['Reason for discharge as Inpatient'], axis=1).copy()
        y = data['Reason for discharge as Inpatient'].copy()
        y = y.astype(bool)

        # 4) Split merged_data_no_td into training, validation and testing with replacement
        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # 5) Define the names of the evaluation metrics used to train the models using randomized grid search, load the
        # models for one metric at a time and place them all together in a single dataframe
        scoring_names = ['error', 'logloss', 'auc', 'aucpr', 'map', 'f1-score', 'balanced accuracy', 'precision', 'recall',
                         'mcc']

        # Initialize dataframe for all evaluation metrics used on the validation set and on the test set for all
        # approaches
        performance_dataframe_val = pd.DataFrame()
        performance_dataframe_test = pd.DataFrame()

        for approach in [1, 2, 3, 4, 5, 6]:
            for i in range(0, len(scoring_names)):
                # Val set
                #data_path_val = r'\Val set\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(int(100*th)) + '.csv'

                if approach == 1 and datasets == 0:
                    data_path_val = r'\All patients\Val set\Results 2\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(int(100 * th)) + '.csv'
                else:
                    data_path_val = r'\Val set\Results 2\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(int(100*th)) + '.csv'

                temp_data_val = pd.read_csv(folder + r'\Approach ' + str(approach) + data_path_val, index_col=0, low_memory=False)
                # Sort by the indices
                temp_data_val = temp_data_val.sort_index()
                # Append data
                performance_dataframe_val = performance_dataframe_val.append(temp_data_val)

                # Test set
                #data_path_test = r'\Test set\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(int(100*th)) + '.csv'

                if approach == 1 and datasets == 0:
                    data_path_test = r'\All patients\Test set\Results 2\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(int(100 * th)) + '.csv'
                else:
                    data_path_test = r'\Test set\Results 2\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(int(100*th)) + '.csv'
                temp_data_test = pd.read_csv(folder + r'\Approach ' + str(approach) + data_path_test, index_col=0, low_memory=False)
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
        #performance_metrics_cols_test = ['Test - Average Precision', 'Test - Precision', 'Test - Accuracy',
        #                                 'Test - Recall', 'Test - F-score', 'Test - MCC', 'Test - AUC']
        for i in performance_metrics_cols_val:
            performance_dataframe_val.loc[:, i] = round(performance_dataframe_val.loc[:, i], 2)
        #for i in performance_metrics_cols_test:
        #    performance_dataframe_test.loc[:, i] = round(performance_dataframe_test.loc[:, i], 2)

        performance_dataframe_val = performance_dataframe_val.drop_duplicates(subset=performance_metrics_cols_val,
                                                                              keep='first')
        performance_dataframe_test = performance_dataframe_test.loc[performance_dataframe_val.index.values, :]


        # Define data path to store figure
        data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Test\Temp'
        #data_path = folder + r'\Approach 1\Visual Results'

        # Scatter plot multi model for the validation set and get the best models
        if datasets == 1:
            model_ids = plotfun.plot_multi_model_performance_metrics(performance_dataframe_val, 'Val', data_path, th,
                                                                     precision_th=0.45, rec_th=0.5, f1_th=0.5,
                                                                     mcc_th=0.4, auc_th=0.7, ap_th=0.4)
        if datasets == 0:
            model_ids = plotfun.plot_multi_model_performance_metrics(performance_dataframe_val, 'Val', data_path, th,
                                                                     rec_th=0.35)

        # Keep the best models from the test models
        performance_dataframe_test = performance_dataframe_test.loc[model_ids, :]

        # Make the same plot for the test set
        plotfun.plot_best_models_performance_metrics(performance_dataframe_test, 'Test', data_path, th)

        # Define models to build
        models = [205, 924, 1336, 5070, 8133, 9879]
        boost_rounds = 1000
        stopping_rounds = 50
        eval_results = dict()
        params = {'max_depth': 6,
                  'min_child_weight': 5,
                  'eta': 0.05,
                  'subsample': 1,
                  'colsample_bytree': 0.5,
                  'scale_pos_weight': 2,
                  'objective': 'binary:logistic',
                  'seed': 42,
                  'disable_default_eval_metric': False
                  }

        # Create a dictionary that has as key the eval metric name and as value the eval function
        eval_metric_dict = {'error': 'error', 'aucpr': 'aucpr', 'logloss': 'logloss', 'map': 'map', 'auc': 'auc',
                            'f1-score': procfun.xgb_f1, 'balanced accuracy': procfun.xgb_balanced_accuracy,
                            'precision': procfun.xgb_precision, 'recall': procfun.xgb_recall, 'mcc': procfun.xgb_mcc}

        for model_id in models:
            # Get model params
            best_params = performance_dataframe_test.loc[model_id, 'Best Params']
            best_params = best_params.replace('[', '')
            best_params = best_params.replace(']', '')
            best_params = best_params.split(',')
            params['max_depth'] = int(best_params[0])
            params['min_child_weight'] = int(best_params[1])
            params['eta'] = float(best_params[2])
            params['subsample'] = float(best_params[3])
            params['colsample_bytree'] = float(best_params[4])
            params['scale_pos_weight'] = float(best_params[5])

            # Get model evaluation metric
            eval_metric = performance_dataframe_test.loc[model_id, 'Eval Metric']
            if eval_metric in ['f1-score', 'precision', 'balanced accuracy', 'mcc', 'recall']:
                params['disable_default_eval_metric'] = True
                params['eval_metric'] = None
                # Build model
                clf_xgb = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=boost_rounds,
                    early_stopping_rounds=stopping_rounds,
                    evals=[(dtrain, "Train"), (dval, "Val")],
                    evals_result=eval_results,
                    feval=eval_metric_dict[eval_metric]
                )
            else:
                params['disable_default_eval_metric'] = False
                params['eval_metric'] = eval_metric
                # Build model
                clf_xgb = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=boost_rounds,
                    early_stopping_rounds=stopping_rounds,
                    evals=[(dtrain, "Train"), (dval, "Val")],
                    evals_result=eval_results,
                )

            # Plot the learning curve
            plotfun.plot_learning_curve_xgb(eval_results, eval_metric, data_path, model_id=model_id,
                                            eval_set_names=['Train', 'Val'])

            # Plot the confusion matrix
            plotfun.plot_confusion_matrix(clf_xgb, dtest, y_test, model_id, data_path)

            # Plot the ROC curve
            plotfun.plot_roc_curve(clf_xgb, dtest, y_test, model_id, data_path)

            # Plot the feature importance barplot
            plotfun.plot_feature_importance_bar(clf_xgb, model_id, data_path)



'''
Test
'''

# Modules
import pandas as pd
import numpy as np
from importlib import reload
import xgboost as xgb
from sklearn.model_selection import train_test_split
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun

merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_after_comorbidities(2)
data = merged_data_no_td.copy()

data = data.drop(['ID', 'COVID diagnosis during admission', 'Date of Admission as Inpatient'], axis=1)
# Replace certain unwanted string characters
data.columns = data.columns.str.replace(',', '')
data.columns = data.columns.str.replace('[', '(')
data.columns = data.columns.str.replace(']', ')')

# Keep only adult people
data = data[data['Age'] >= 18]

th=0.85

data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

# 3) Split into independent (X) and dependent variables (y)
X = data.drop(['Reason for discharge as Inpatient'], axis=1).copy()
y = data['Reason for discharge as Inpatient'].copy()
y = y.astype(bool)

# 4) Split merged_data_no_td into training, validation and testing with replacement
# Split into train and val+test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
# Split val+test into val and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

# Create the DMatrix of the datasets
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)
dtest = xgb.DMatrix(data=X_test, label=y_test)

boost_rounds = 1000
stopping_rounds = 50
eval_results = dict()
params = {'max_depth': 6,
          'min_child_weight': 5,
          'eta': 0.05,
          'subsample': 1,
          'colsample_bytree': 0.5,
          'scale_pos_weight': 2,
          'objective': 'binary:logistic',
          'seed': 42,
          'tree_method': 'hist',
          'grow_policy': 'depthwise',
          'disable_default_eval_metric': False
          }

params['max_depth'] = 8
params['min_child_weight'] = 3
params['eta'] = 0.1
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['scale_pos_weight'] = 1
params['eval_metric'] = 'error'

clf_xgb = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=boost_rounds,
    early_stopping_rounds=stopping_rounds,
    evals=[(dtrain, "Train"), (dval, "Val")],
    evals_result=eval_results,
)
