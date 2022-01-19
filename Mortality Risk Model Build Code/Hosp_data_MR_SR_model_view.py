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
import pickle

'''
GENERAL STEPS
'''
# 1) Load specific datasets
data_name = 'merged data no td'
category = 4
sub_category = 'Diabetes'
th = 0.85
approach = 1

if category in [3, 4]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)
    data1 = data[0]
    data2 = data[1]
elif category in [1, 2]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)

# Split the data into training for the k-fold cv and into test using exactly the same split format as the one
# used at the grid searches performed to find the best models.
# Split data to features and class
X = data2.drop(['Reason for discharge as Inpatient'], axis=1).copy()
y = data2['Reason for discharge as Inpatient'].copy()
y = y.astype(bool)
# Split into train and val+test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
# Split val+test into val and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp,
                                                test_size=0.5)

# Create the DMatrix of the datasets
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# 2) Define folders where grid search models are stored and get all models
folder_val = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data no td com\Tuning 3\Approach 1\Diabetes Only\Negative Patients\Val Set'
folder_test = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data no td com\Tuning 3\Approach 1\Diabetes Only\Negative Patients\Test Set'

performance_dataframe_val, performance_dataframe_test = datafun.load_gridsearch_models(folder_val, folder_test, th)

# 3) Define models to build
boost_rounds = 1000
stopping_rounds = 50
models = [207]

# Create a dictionary that has as key the eval metric name and as value the eval function
eval_metric_dict = {'error': 'error', 'aucpr': 'aucpr', 'logloss': 'logloss', 'map': 'map', 'auc': 'auc',
                    'f1-score': modelfun.xgb_f1, 'balanced accuracy': modelfun.xgb_balanced_accuracy,
                    'precision': modelfun.xgb_precision, 'recall': modelfun.xgb_recall, 'mcc': modelfun.xgb_mcc}

data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Test\Method 3\merged data no td no severity\Diabetes\Negative Patients'

for model_id in models:
    # Get model params
    best_params = performance_dataframe_test.loc[model_id, 'Best Params']
    best_params = best_params.replace('[', '')
    best_params = best_params.replace(']', '')
    best_params = best_params.split(',')

    params = {'max_depth': best_params[0],
              'min_child_weight': best_params[1],
              'eta': float(best_params[2]),
              'subsample': float(best_params[3]),
              'colsample_bytree': float(best_params[4]),
              'scale_pos_weight': float(best_params[5]),
              'objective': 'binary:logistic',
              'seed': 42,
              'tree_method': 'hist',
              'grow_policy': 'depthwise',
              'disable_default_eval_metric': False
              }

    # Initialize eval_results
    eval_results = {}

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

    # Store model
    filename = r'\xgboost_model_' + str(model_id) + '.pkl'
    data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Final Models chosen\Merged data no td\No Severity\Negative Diabetes'
    pickle.dump(clf_xgb, open(data_path + filename, "wb"))
