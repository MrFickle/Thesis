# Modules
import pandas as pd
import numpy as np
from importlib import reload
import data_exploratory_functions as myfun
import xgboost as xgb
from sklearn.model_selection import train_test_split
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun
from sklearn.utils import resample
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, average_precision_score, \
    matthews_corrcoef
import pickle
from xgboost.sklearn import XGBClassifier
import random
import time


'''
GENERAL STEPS
'''
# 1) Load specific datasets
data_name = 'merged data no td'
category = 1
sub_category = None
th = 0.85

if category in [3, 4]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)
    data1 = data[0]
    data2 = data[1]
elif category in [1, 2]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)

# 2) Define model params, scoring function and scoring function name
params = {'max_depth': 9,
          'min_child_weight': 5,
          'eta': 0.05,
          'subsample': 0.8,
          'colsample_bytree': 0.6,
          'scale_pos_weight': 1,
          'objective': 'binary:logistic',
          'seed': 42
          }

scoring_function = 'error'
scoring_name = 'error'

# 3) Split the data into training for the k-fold cv and into test using exactly the same split format as the one
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
# Now keep the test set and merge the validation set with the training set and pass it on to the k-fold cv
X_train = X_train.append(X_val)
y_train = y_train.append(y_val)
data_train = X_train
data_train['Reason for discharge as Inpatient'] = y_train

# 3) Perform 9-fold cross validation
performance_nfold_cv_val, performance_nfold_cv_test = \
    modelfun.nfold_xgb_cv(data_train, X_test, y_test, params, scoring_function, scoring_name)


