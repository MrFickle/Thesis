# Modules
import pandas as pd
import numpy as np
from importlib import reload
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import random


'''
GENERAL STEPS
'''
# 1) Load specific datasets
data_name = 'merged data no td'
category = 1
sub_category = 'None'
th = 0.85

if category in [3, 4]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)
    data1 = data[0]
    data2 = data[1]
elif category in [1, 2]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)

# Split the data into training for the k-fold cv and into test using exactly the same split format as the one
# used at the grid searches performed to find the best models.
# Split data to features and class
X = data.drop(['Reason for discharge as Inpatient'], axis=1).copy()
y = data['Reason for discharge as Inpatient'].copy()
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
data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data no td com\Tuning 3\Approach 1\Diabetes Only\Negative Patients\CV Results'

mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val, \
std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test, \
performance_dataframe_val, performance_dataframe_test = datafun.load_cv_models(data_path, th)

# 3) Define models to build
boost_rounds = 1000
stopping_rounds = 50
model_id = 207

# Create a dictionary that has as key the eval metric name and as value the eval function
eval_metric_dict = {'error': 'error', 'aucpr': 'aucpr', 'logloss': 'logloss', 'map': 'map', 'auc': 'auc',
                    'f1-score': modelfun.xgb_f1, 'balanced accuracy': modelfun.xgb_balanced_accuracy,
                    'precision': modelfun.xgb_precision, 'recall': modelfun.xgb_recall, 'mcc': modelfun.xgb_mcc}

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

# Store model
filename = r'\xgboost_model_' + str(model_id) + '.pkl'
data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Final Models chosen\Merged data no td\No Severity\Negative Diabetes'
pickle.dump(clf_xgb, open(data_path + filename, "wb"))

# Load model
xgb_model_loaded = pickle.load(open(data_path + filename, "rb"))
# Patients to test
test_patients_loc = [1227, 805, 91, 856, 903, 358, 575, 990, 202, 185]
test_patients_X = X_test.loc[test_patients_loc, :].copy()
test_patients_y = y_test.loc[test_patients_loc].copy()

test_patients_dtest = xgb.DMatrix(data=test_patients_X, label=test_patients_y)

predictions = xgb_model_loaded.predict(test_patients_dtest, ntree_limit=xgb_model_loaded.best_iteration + 1)
test_patients_predictions = pd.DataFrame(data={'Patient ID': test_patients_loc, 'Model predictions': 100*np.round(predictions, 2), 'Real Values': test_patients_y})


