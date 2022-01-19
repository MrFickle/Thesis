"""
This script contains functions that are used only for processing data.
"""

# Modules
import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats import kstest
import scipy.stats as stats
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_proc_funs as procfun
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, average_precision_score, \
    matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import random


# Create a custom evaluation metric function for xgboost using the precision metric:
def xgb_precision(y_pred, y_true):
    y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    return 'precision', -precision_score(y_true.get_label().astype(bool), np.round(y_pred).astype(bool))


# Create a custom evaluation metric function for xgboost using the balanced accuracy sklearn metric:
def xgb_balanced_accuracy(y_pred, y_true):
    y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
    return 'balanced_accuracy', -balanced_accuracy_score(y_true.get_label().astype(bool), np.round(y_pred).astype(bool))


# Create a custom evaluation metric function for xgboost using the f1 score sklearn metric:
def xgb_f1(y_pred, y_true):
    y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
    return 'f1_score', -f1_score(y_true.get_label().astype(bool), np.round(y_pred).astype(bool), zero_division=0)


# Create a custom evaluation metric function for xgboost using the recall sklearn metric:
def xgb_recall(y_pred, y_true):
    y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
    return 'recall', -recall_score(y_true.get_label().astype(bool), np.round(y_pred).astype(bool))


# Create a custom evaluation metric function for xgboost using the mcc sklearn metric:
def xgb_mcc(y_pred, y_true):
    y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
    return 'mcc', -matthews_corrcoef(y_true.get_label().astype(bool), np.round(y_pred).astype(bool))


def modelfit(xgb1, X_train, y_train, dtrain, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = xgb1.get_xgb_params()
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        xgb1.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    xgb1.fit(X_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = xgb1.predict(X_train)
    dtrain_predprob = xgb1.predict_proba(X_train)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(y_train.values, dtrain_predprob))
    print('Precision (Train): %f' % precision_score(y_train.values, dtrain_predictions))

    return cvresult


# Create a function that receives a set of gridsearch params and returns the best model based on a specific eval metric
# using cross validation
def find_best_model(gridsearch_params, gridsearch_param_names, params, dtrain, dval, boost_rounds, stopping_rounds,
                    cv_metric, cv_folds=5):
    # Initialize best score and best params
    if cv_metric == 'std':
        best_score = 1
    elif cv_metric == 'mean':
        best_score = 0
    best_params = []
    best_rounds = 0

    # Build model using gridsearch
    for param_values in gridsearch_params:
        for i in range(0, len(gridsearch_param_names)):
            if len(gridsearch_param_names) == 1:
                params[gridsearch_param_names[i]] = param_values
            else:
                params[gridsearch_param_names[i]] = param_values[i]

        # Create model using cross validation
        cvresult = xgb.cv(params, dtrain, num_boost_round=boost_rounds, nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=stopping_rounds)

        # Get best score based on validation set
        if cv_metric == 'std':
            model_best_score = abs(cvresult.iloc[cvresult.shape[0]-1, 3])
        elif cv_metric == 'mean':
            model_best_score = abs(cvresult.iloc[cvresult.shape[0] - 1, 2])
        # Get best rounds
        model_best_rounds = cvresult.shape[0]
        if cv_metric == 'std':
            if model_best_score < best_score:
                best_score = model_best_score
                if len(gridsearch_param_names) == 1:
                    best_params = param_values
                else:
                    best_params = [i for i in param_values]
                best_rounds = model_best_rounds
        elif cv_metric == 'mean':
            if model_best_score > best_score:
                best_score = model_best_score
                if len(gridsearch_param_names) == 1:
                    best_params = param_values
                else:
                    best_params = [i for i in param_values]
                best_rounds = model_best_rounds

    # Update the params using the best params found
    for i in range(0, len(gridsearch_param_names)):
        if len(gridsearch_param_names) == 1:
            params[gridsearch_param_names[i]] = best_params
        else:
            params[gridsearch_param_names[i]] = best_params[i]

    # Return results
    return best_score, best_params, best_rounds, params


# Create a function that calculates the performance metrics and returns a dataframe containing their values
def get_performance_metrics(params, dtrain, dval, dtest, y_test, boost_rounds, stopping_rounds, scoring_names,
                            best_score):
    # Get the best params
    best_params = [params['max_depth'], params['min_child_weight'], params['eta'], params['subsample'],
                   params['colsample_bytree'], params['scale_pos_weight'], params['min_split_loss'],
                   params['reg_lambda'], params['reg_alpha']]

    # Initialize dataframe
    performance_dataframe = pd.DataFrame(columns=['Eval Metric', 'Best Eval Metric Score', 'Average Precision',
                                                  'Precision', 'Accuracy', 'Recall', 'F-score', 'MCC',
                                                  'AUC on Test Set', 'Best Params'],
                                         index=range(0, len(scoring_names)))

    # Build model
    clf_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=boost_rounds,
        early_stopping_rounds=stopping_rounds,
        evals=[(dval, "Val")]
    )

    # Make prediction on test set
    y_pred = clf_xgb.predict(dtest, ntree_limit=clf_xgb.best_iteration + 1)
    # Average precision = AP = sum_n((recall_n - recall_n-1)/Pn)
    ap_score = average_precision_score(y_test.values, y_pred)
    # Precision score
    check2 = y_pred.round()
    prec_score = precision_score(y_test.values, check2)
    # Recall score
    rec_score = recall_score(y_test.values, check2)
    # Accuracy score
    ac_score = accuracy_score(y_test.values, check2)
    # F-score
    f_score = 2 * prec_score * rec_score / (prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_test.values, check2)
    # AUC
    fpr, tpr, _ = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)

    for i in range(0, len(scoring_names)):
        # Store the results in the performance dataframe
        performance_dataframe.loc[i, 'Eval Metric'] = scoring_names[i]
        performance_dataframe.loc[i, 'Best Eval Metric Score'] = clf_xgb.best_score
        performance_dataframe.loc[i, 'Average Precision'] = ap_score
        performance_dataframe.loc[i, 'Precision'] = prec_score
        performance_dataframe.loc[i, 'Accuracy'] = ac_score
        performance_dataframe.loc[i, 'Recall'] = rec_score
        performance_dataframe.loc[i, 'F-score'] = f_score
        performance_dataframe.loc[i, 'MCC'] = mcc
        performance_dataframe.loc[i, 'AUC on Test Set'] = roc_auc
        performance_dataframe.loc[i, 'Best Params'] = best_params
        performance_dataframe = performance_dataframe.sort_values(by='Precision', ascending=False)

    return performance_dataframe


# Create a function that creates an xgboost model given specific params and calculates its performance metrics on the
# validation and test sets and stores the results in a dataframe given for that purpose.
def single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain, dval, dtest, y_val,
                                                y_test, scoring_function, performance_dataframe_val,
                                                performance_dataframe_test, scoring_function_name, count):
    # Check whether the scoring function is custom or not
    if scoring_function_name in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
        clf_xgb = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=boost_rounds,
            early_stopping_rounds=stopping_rounds,
            evals=[(dval, "Val")],
            feval=scoring_function
        )
    else:
        clf_xgb = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=boost_rounds,
            early_stopping_rounds=stopping_rounds,
            evals=[(dval, "Val")]
        )

    # Get best score
    model_best_score = abs(clf_xgb.best_score)
    # Get best rounds
    model_best_rounds = clf_xgb.best_iteration + 1
    best_params = [params['max_depth'], params['min_child_weight'], params['eta'], params['subsample'],
                   params['colsample_bytree'], params['scale_pos_weight']]

    # Make prediction on val set
    y_pred = clf_xgb.predict(dval, ntree_limit=clf_xgb.best_iteration + 1)
    # Average precision = AP = sum_n((recall_n - recall_n-1)/Pn)
    ap_score = average_precision_score(y_val.values, y_pred)
    # Precision score
    check2 = y_pred.round()
    prec_score = precision_score(y_val.values, check2)
    # Recall score
    rec_score = recall_score(y_val.values, check2)
    # Accuracy score
    ac_score = accuracy_score(y_val.values, check2)
    # F-score
    f_score = 2 * prec_score * rec_score / (prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_val.values, check2)
    # AUC
    fpr, tpr, _ = roc_curve(y_val.values, y_pred)
    roc_auc = auc(fpr, tpr)

    # 10) Store the results in the performance dataframe
    performance_dataframe_val.loc[count, 'Eval Metric'] = scoring_function_name
    performance_dataframe_val.loc[count, 'Val - Best Eval Metric Score'] = model_best_score
    performance_dataframe_val.loc[count, 'Val - Average Precision'] = ap_score
    performance_dataframe_val.loc[count, 'Val - Precision'] = prec_score
    performance_dataframe_val.loc[count, 'Val - Accuracy'] = ac_score
    performance_dataframe_val.loc[count, 'Val - Recall'] = rec_score
    performance_dataframe_val.loc[count, 'Val - F-score'] = f_score
    performance_dataframe_val.loc[count, 'Val - MCC'] = mcc
    performance_dataframe_val.loc[count, 'Val - AUC'] = roc_auc
    performance_dataframe_val.loc[count, 'Best Params'] = best_params
    performance_dataframe_val.loc[count, 'Best Round'] = model_best_rounds

    # Make prediction on test set
    y_pred = clf_xgb.predict(dtest, ntree_limit=clf_xgb.best_iteration + 1)
    # Average precision = AP = sum_n((recall_n - recall_n-1)/Pn)
    ap_score = average_precision_score(y_test.values, y_pred)
    # Precision score
    check2 = y_pred.round()
    prec_score = precision_score(y_test.values, check2)
    # Recall score
    rec_score = recall_score(y_test.values, check2)
    # Accuracy score
    ac_score = accuracy_score(y_test.values, check2)
    # F-score
    f_score = 2 * prec_score * rec_score / (prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_test.values, check2)
    # AUC
    fpr, tpr, _ = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)

    performance_dataframe_test.loc[count, 'Eval Metric'] = scoring_function_name
    performance_dataframe_test.loc[count, 'Val - Best Eval Metric Score'] = model_best_score
    performance_dataframe_test.loc[count, 'Test - Average Precision'] = ap_score
    performance_dataframe_test.loc[count, 'Test - Precision'] = prec_score
    performance_dataframe_test.loc[count, 'Test - Accuracy'] = ac_score
    performance_dataframe_test.loc[count, 'Test - Recall'] = rec_score
    performance_dataframe_test.loc[count, 'Test - F-score'] = f_score
    performance_dataframe_test.loc[count, 'Test - MCC'] = mcc
    performance_dataframe_test.loc[count, 'Test - AUC'] = roc_auc
    performance_dataframe_test.loc[count, 'Best Params'] = best_params
    performance_dataframe_test.loc[count, 'Best Round'] = model_best_rounds

    count = count + 1

    # Return dataframes and counter
    return performance_dataframe_val, performance_dataframe_test, count


# Create a function that receives the data to be used for training the xgboost model and using randomized gridsearch
# based on the approach desired, it returns the results of every single model for that approach:
# Possible approaches:
# Approach 1: General param optimization with missing values
# Approach 2: General param optimization with filling missing values using imputer
# Approach 3: General param optimization with undersampling majority class
# Approach 4: General param optimization with undersampling majority class and filling missing values using imputer
# Approach 5: General param optimization with SMOTE
# Approach 6: General param optimization with SMOTE + undersampling majority class
# Approach 7: General param optimization with missing values creating N models, where for each model the
# positive/negative ratio = 1 and the final outcome is a mean prediction.
def choose_grid_search_ML_approach(data, approach, scoring_functions, scoring_names, gridsearch_params, smote_params,
                                   boost_rounds, stopping_rounds, folder, th):

    # Split into independent (X) and dependent variables (y)
    X = data.drop(['Reason for discharge as Inpatient'], axis=1).copy()
    y = data['Reason for discharge as Inpatient'].copy()
    y = y.astype(bool)

    if approach == 1:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round'],
                                                     index=range(0, len(gridsearch_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=range(0, len(gridsearch_params)))

            count = 0
            # Build model using gridsearch
            for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                # Update params
                params['max_depth'] = max_depth
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta
                params['subsample'] = subsample
                params['colsample_bytree'] = colsample_bytree
                params['scale_pos_weight'] = scale_pos_weight

                performance_dataframe_val, performance_dataframe_test, count = \
                    single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain,
                                                                        dval, dtest, y_val, y_test,
                                                                        scoring_functions[i],
                                                                        performance_dataframe_val,
                                                                        performance_dataframe_test,
                                                                        scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 2:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Fill missing values for each set
        X_train, X_val, X_test = procfun.fill_train_valid_test_sets(X_train, y_train, X_val, y_val, X_test, y_test)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round'],
                                                     index=range(0, len(gridsearch_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=range(0, len(gridsearch_params)))

            count = 0
            # Build model using gridsearch
            for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                # Update params
                params['max_depth'] = max_depth
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta
                params['subsample'] = subsample
                params['colsample_bytree'] = colsample_bytree
                params['scale_pos_weight'] = scale_pos_weight

                performance_dataframe_val, performance_dataframe_test, count = \
                    single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain,
                                                                dval, dtest, y_val, y_test,
                                                                scoring_functions[i],
                                                                performance_dataframe_val,
                                                                performance_dataframe_test,
                                                                scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 3:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round', 'Class Ratio'],
                                                     index=range(0, len(gridsearch_params)*len(smote_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round',
                                                               'Class Ratio'],
                                                      index=range(0, len(gridsearch_params)*len(smote_params)))

            count = 0

            for class_ratio in smote_params:
                # Merge the X_train and y_train datasets in order to downsample the majority class afterwards
                XY_train = X_train.copy()
                XY_train['Class'] = y_train.copy()

                # Downsample the majority class to reach the class_ratio defined.
                data_minority = XY_train[XY_train['Class'] == 1]
                data_majority = XY_train[XY_train['Class'] == 0]
                data_majority_downsampled = resample(data_majority, replace=False,
                                                     n_samples=round(data_minority.shape[0] / class_ratio),
                                                     random_state=42)

                # Combine minority class with downsampled majority class
                XY_train = data_minority.append(data_majority_downsampled)

                # Split again into X and y
                X_train3 = XY_train.drop('Class', axis=1).copy()
                y_train3 = XY_train['Class'].copy()

                # Convert train dataset to Dmatrix
                dtrain3 = xgb.DMatrix(X_train3, label=y_train3)

                # Build model using gridsearch
                for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                    # Update params
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['eta'] = eta
                    params['subsample'] = subsample
                    params['colsample_bytree'] = colsample_bytree
                    params['scale_pos_weight'] = scale_pos_weight

                    performance_dataframe_val.loc[count, 'Class Ratio'] = class_ratio
                    performance_dataframe_test.loc[count, 'Class Ratio'] = class_ratio

                    performance_dataframe_val, performance_dataframe_test, count = \
                        single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain3,
                                                                    dval, dtest, y_val, y_test,
                                                                    scoring_functions[i],
                                                                    performance_dataframe_val,
                                                                    performance_dataframe_test,
                                                                    scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 4:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Fill missing values for each set
        X_train, X_val, X_test = procfun.fill_train_valid_test_sets(X_train, y_train, X_val, y_val, X_test, y_test)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round', 'Class Ratio'],
                                                     index=range(0, len(gridsearch_params) * len(smote_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round',
                                                               'Class Ratio'],
                                                      index=range(0, len(gridsearch_params) * len(smote_params)))

            count = 0

            for class_ratio in smote_params:
                # Merge the X_train and y_train datasets in order to downsample the majority class afterwards
                XY_train = X_train.copy()
                XY_train['Class'] = y_train.copy()

                # Downsample the majority class to reach the class_ratio defined.
                data_minority = XY_train[XY_train['Class'] == 1]
                data_majority = XY_train[XY_train['Class'] == 0]
                data_majority_downsampled = resample(data_majority, replace=False,
                                                     n_samples=round(data_minority.shape[0] / class_ratio),
                                                     random_state=42)

                # Combine minority class with downsampled majority class
                XY_train = data_minority.append(data_majority_downsampled)

                # Split again into X and y
                X_train3 = XY_train.drop('Class', axis=1).copy()
                y_train3 = XY_train['Class'].copy()

                # Convert train dataset to Dmatrix
                dtrain3 = xgb.DMatrix(X_train3, label=y_train3)

                # Build model using gridsearch
                for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                    # Update params
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['eta'] = eta
                    params['subsample'] = subsample
                    params['colsample_bytree'] = colsample_bytree
                    params['scale_pos_weight'] = scale_pos_weight

                    performance_dataframe_val.loc[count, 'Class Ratio'] = class_ratio
                    performance_dataframe_test.loc[count, 'Class Ratio'] = class_ratio

                    performance_dataframe_val, performance_dataframe_test, count = \
                        single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain3,
                                                                    dval, dtest, y_val, y_test,
                                                                    scoring_functions[i],
                                                                    performance_dataframe_val,
                                                                    performance_dataframe_test,
                                                                    scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 5:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Fill missing values for each set
        X_train2, X_val2, X_test2 = procfun.fill_train_valid_test_sets(X_train, y_train, X_val, y_val, X_test, y_test)

        # Create the DMatrix of the datasets
        dval2 = xgb.DMatrix(data=X_val2, label=y_val)
        dtest2 = xgb.DMatrix(data=X_test2, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round', 'SMOTE Params'],
                                                     index=range(0, len(gridsearch_params) * len(smote_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round',
                                                               'SMOTE Params'],
                                                      index=range(0, len(gridsearch_params) * len(smote_params)))

            count = 0

            for smote_ratio, smote_neighbours in smote_params:
                # Oversample the minority class using SMOTE to reach the smote_ratio defined.
                oversample_minority = SMOTE(sampling_strategy=smote_ratio, k_neighbors=smote_neighbours,
                                            random_state=42)
                X_train3, y_train3 = oversample_minority.fit_resample(X_train2, y_train)

                # Convert train dataset to Dmatrix
                dtrain3 = xgb.DMatrix(X_train3, label=y_train3)

                # Build model using gridsearch
                for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                    # Update params
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['eta'] = eta
                    params['subsample'] = subsample
                    params['colsample_bytree'] = colsample_bytree
                    params['scale_pos_weight'] = scale_pos_weight

                    performance_dataframe_val.loc[count, 'SMOTE Params'] = [smote_ratio, smote_neighbours]
                    performance_dataframe_test.loc[count, 'SMOTE Params'] = [smote_ratio, smote_neighbours]

                    performance_dataframe_val, performance_dataframe_test, count = \
                        single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain3,
                                                                    dval2, dtest2, y_val, y_test,
                                                                    scoring_functions[i],
                                                                    performance_dataframe_val,
                                                                    performance_dataframe_test,
                                                                    scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 6:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Fill missing values for each set
        X_train2, X_val2, X_test2 = procfun.fill_train_valid_test_sets(X_train, y_train, X_val, y_val, X_test, y_test)

        # Create the DMatrix of the datasets
        dval2 = xgb.DMatrix(data=X_val2, label=y_val)
        dtest2 = xgb.DMatrix(data=X_test2, label=y_test)

        # For every evaluation metric create a model for all combinations
        for i in range(0, len(scoring_functions)):
            if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'disable_default_eval_metric': True
                          }
            else:
                # Initialize params
                params = {'objective': 'binary:logistic',
                          'seed': 42,
                          'tree_method': 'hist',
                          'grow_policy': 'depthwise',
                          'eval_metric': scoring_functions[i]
                          }

            performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                              'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Params', 'Best Round', 'SMOTE Params'],
                                                     index=range(0, len(gridsearch_params) * len(smote_params)))

            performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round',
                                                               'SMOTE Params'],
                                                      index=range(0, len(gridsearch_params) * len(smote_params)))

            count = 0

            for smote_ratio, smote_neighbours, class_ratio in smote_params:
                # Oversample the minority class using SMOTE to reach the smote_ratio defined.
                oversample_minority = SMOTE(sampling_strategy=smote_ratio, k_neighbors=smote_neighbours,
                                            random_state=42)
                X_train3, y_train3 = oversample_minority.fit_resample(X_train2, y_train)

                # Merge the X_train and y_train datasets in order to downsample the majority class afterwards
                XY_train = X_train3.copy()
                XY_train['Class'] = y_train3

                # Downsample the majority class to reach the class_ratio defined.
                data_minority = XY_train[XY_train['Class'] == 1]
                data_majority = XY_train[XY_train['Class'] == 0]
                data_majority_downsampled = resample(data_majority, replace=False,
                                                     n_samples=round(data_minority.shape[0] / class_ratio),
                                                     random_state=42)

                # Combine minority class with downsampled majority class
                XY_train = data_minority.append(data_majority_downsampled)

                # Split again into X and y
                X_train3 = XY_train.drop('Class', axis=1).copy()
                y_train3 = XY_train['Class'].copy()

                # Convert train dataset to Dmatrix
                dtrain3 = xgb.DMatrix(X_train3, label=y_train3)

                # Build model using gridsearch
                for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                    # Update params
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['eta'] = eta
                    params['subsample'] = subsample
                    params['colsample_bytree'] = colsample_bytree
                    params['scale_pos_weight'] = scale_pos_weight

                    performance_dataframe_val.loc[count, 'SMOTE Params'] = [smote_ratio, smote_neighbours, class_ratio]
                    performance_dataframe_test.loc[count, 'SMOTE Params'] = [smote_ratio, smote_neighbours, class_ratio]

                    performance_dataframe_val, performance_dataframe_test, count = \
                        single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain3,
                                                                    dval2, dtest2, y_val, y_test,
                                                                    scoring_functions[i],
                                                                    performance_dataframe_val,
                                                                    performance_dataframe_test,
                                                                    scoring_names[i], count)

            data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
            performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

            data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
            filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                int(100 * th)) + '.csv'
            performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
            performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)

    elif approach == 7:

        # Split into train and val+test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
        # Split val+test into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

        # Create the DMatrix for the test and val datasets
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # Find the percentage of positive patients in the dataset
        pos_ratio = y_train.value_counts()[1]/y_train.shape[0]
        # Get the number of patients that belong in the minority class (positive)
        pos_num = y_train.value_counts()[1]
        # Merge the X_train and y_train data in X_train
        X_train['Class'] = y_train
        # Separate the positive from the negative patients
        X_train_pos = X_train[X_train['Class'] == 1].copy()
        X_train_neg = X_train[X_train['Class'] == 0].copy()
        # Initialize a list that will contain the new training datasets with equal class ratios
        train_list = list()
        # Start creating new training datasets with equal class ratios by taking the same positive patients along with
        # random negative patients without replacement from the X_train_neg data. If the X_train_neg data has less
        # patients than the ones required to be sampled, then take the last remaining patients and random sample the
        # rest from the X_train data.
        while ~X_train_neg.empty:
            if X_train_neg.shape[0] >= pos_num:
                # Get the indices of X_train_neg
                neg_indices = X_train_neg.index.values
                # Get random negative patients
                random.seed(42)
                random_patients = random.sample(neg_indices, pos_num)
                # Create the new training dataset
                new_train = X_train_pos.copy()
                new_train = new_train.append(X_train_neg.loc[random_patients, :])
                # Remove the random patients from X_train_neg
                X_train_neg = X_train_neg.drop(labels=random_patients, axis=0)
                # Place the new_train data at the list
                train_list.append(new_train)

            # If the X_train_neg no longer has sufficient number of negative patients, then get the remaining patients
            # and random sample the final ones from the X_train data without repetition.
            else:
                # Create the new training dataset
                new_train = X_train_pos.copy()
                new_train = new_train.append(X_train_neg)
                # Empty the X_train_neg data
                X_train_neg = pd.DataFrame()
                # Get the number of required patients needed
                required_patients = pos_num - new_train.shape[0]
                # Random sample the required patients from the X_train data and place them in the new_train data
                random.seed(42)
                random_patients = random.sample(X_train.index.values, required_patients)
                new_train = new_train.append(X_train.loc[random_patients, :])
                # Place the new_train data at the list
                train_list.append(new_train)

        # Now for every new_train data, create an XGBoost model:
        for j in range(0, len(train_list)):
            X_train = train_list[j]
            y_train = X_train['Class']
            X_train = X_train.drop(columns='Class')

            # Create the DMatrix for the train dataset
            dtrain = xgb.DMatrix(data=X_train, label=y_train)

            # For every evaluation metric create a model for all combinations
            for i in range(0, len(scoring_functions)):
                if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
                    # Initialize params
                    params = {'objective': 'binary:logistic',
                              'seed': 42,
                              'tree_method': 'hist',
                              'grow_policy': 'depthwise',
                              'disable_default_eval_metric': True
                              }
                else:
                    # Initialize params
                    params = {'objective': 'binary:logistic',
                              'seed': 42,
                              'tree_method': 'hist',
                              'grow_policy': 'depthwise',
                              'eval_metric': scoring_functions[i]
                              }

                performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                  'Val - Average Precision', 'Val - Precision',
                                                                  'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                                  'Val - MCC',
                                                                  'Val - AUC', 'Best Params', 'Best Round'],
                                                         index=range(0, len(gridsearch_params)))

                performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                   'Test - Average Precision', 'Test - Precision',
                                                                   'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                                   'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                          index=range(0, len(gridsearch_params)))

                count = 0
                # Build model using gridsearch
                for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
                    # Update params
                    params['max_depth'] = max_depth
                    params['min_child_weight'] = min_child_weight
                    params['eta'] = eta
                    params['subsample'] = subsample
                    params['colsample_bytree'] = colsample_bytree
                    params['scale_pos_weight'] = scale_pos_weight

                    performance_dataframe_val, performance_dataframe_test, count = \
                        single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain,
                                                                    dval, dtest, y_val, y_test,
                                                                    scoring_functions[i],
                                                                    performance_dataframe_val,
                                                                    performance_dataframe_test,
                                                                    scoring_names[i], count)

                data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
                filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
                    int(100 * th)) + '.csv'
                performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
                performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

                data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
                filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
                    int(100 * th)) + '.csv'
                performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
                performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)


# Create a function that receives a model's params, evaluation metric, evaluation metric name and performs n-fold
# cross validation making predictions
def nfold_xgb_cv(data, X_test, y_test, params, scoring_function, scoring_name, n_folds=9, boost_rounds=1000,
                 stopping_rounds=50):
    # Split the data into stratified n_folds
    index_values_pos = data[data['Reason for discharge as Inpatient'] == 1].index.values
    index_values_neg = data[data['Reason for discharge as Inpatient'] == 0].index.values
    # Class ratio
    pos_ratio = len(index_values_pos) / data.shape[0]

    fold_size = round(data.shape[0]/n_folds)
    fold_size_pos = int(np.round(pos_ratio * fold_size))
    fold_size_neg = int(np.round((1-pos_ratio)*fold_size))
    data_folds = list()
    for i in range(0, n_folds):
        if i < n_folds-1:
            random.seed(42)
            temp_fold_pos = random.sample(index_values_pos.tolist(), fold_size_pos)
            random.seed(42)
            temp_fold_neg = random.sample(index_values_neg.tolist(), fold_size_neg)
            temp_data = data.loc[temp_fold_pos, :]
            temp_data = temp_data.append(data.loc[temp_fold_neg, :])
            data_folds.append(temp_data)
            # Remove used rows
            index_values_pos = np.setdiff1d(index_values_pos, temp_fold_pos)
            index_values_neg = np.setdiff1d(index_values_neg, temp_fold_neg)
        else:
            temp_data = data.loc[index_values_pos, :]
            temp_data = temp_data.append(data.loc[index_values_neg, :])
            data_folds.append(temp_data)

    # Initialize a performance dataframe that contains n_folds rows (one model per fold) and the average score of all
    # models both for the val set and for the test set
    performance_nfold_cv_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                  'Val - Average Precision', 'Val - Precision',
                                                                  'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                                  'Val - MCC',
                                                                  'Val - AUC', 'Best Params', 'Best Round'],
                                                         index=range(0, n_folds+3))
    performance_nfold_cv_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=range(0, n_folds+3))

    if scoring_name in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
        params['disable_default_eval_metric'] = True
        params['eval_metric'] = None
    else:
        params['disable_default_eval_metric'] = False
        params['eval_metric'] = scoring_function

    # Perform n-fold cross validation
    for i in range(0, n_folds):
        data_val = data_folds[i]
        data_train = pd.DataFrame()
        for j in range(0, n_folds):
            if i != j:
                data_train = data_train.append(data_folds[j])
        # Split data to features and class
        # Train
        X_train = data_train.drop(['Reason for discharge as Inpatient'], axis=1).copy()
        y_train = data_train['Reason for discharge as Inpatient'].copy()
        y_train = y_train.astype(bool)

        # Val
        X_val = data_val.drop(['Reason for discharge as Inpatient'], axis=1).copy()
        y_val = data_val['Reason for discharge as Inpatient'].copy()
        y_val = y_val.astype(bool)

        # Create the DMatrix of the datasets
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        # Train model
        performance_nfold_cv_val, performance_nfold_cv_test, count = \
            single_model_performance_metric_calculation(params, boost_rounds, stopping_rounds, dtrain, dval, dtest,
                                                    y_val,
                                                    y_test, scoring_function, performance_nfold_cv_val,
                                                    performance_nfold_cv_test, scoring_name, i)

    # Get average score
    # Val
    scoring_columns_val = ['Val - Best Eval Metric Score', 'Val - Average Precision', 'Val - Precision',
                                                              'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                              'Val - MCC',
                                                              'Val - AUC', 'Best Round']
    for j in scoring_columns_val:
        performance_nfold_cv_val.loc[n_folds, j] = \
            performance_nfold_cv_val.loc[0:n_folds-1, j].mean()
        performance_nfold_cv_val.loc[n_folds+1, j] = \
            performance_nfold_cv_val.loc[0:n_folds-1, j].std()
        performance_nfold_cv_val.loc[n_folds+2, j] = \
            performance_nfold_cv_val.loc[0:n_folds-1, j].sem()

    # Test
    scoring_columns_test = ['Val - Best Eval Metric Score', 'Test - Average Precision', 'Test - Precision',
                                                              'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                              'Test - MCC',
                                                              'Test - AUC', 'Best Round']

    for j in scoring_columns_test:
        performance_nfold_cv_test.loc[n_folds, j] = \
            performance_nfold_cv_test.loc[0:n_folds-1, j].mean()
        performance_nfold_cv_test.loc[n_folds+1, j] = \
            performance_nfold_cv_test.loc[0:n_folds-1, j].std()
        performance_nfold_cv_test.loc[n_folds+2, j] = \
            performance_nfold_cv_test.loc[0:n_folds-1, j].sem()

    # Val
    performance_nfold_cv_val.loc[n_folds, 'Eval Metric'] = scoring_name + ' - mean'
    performance_nfold_cv_val.loc[n_folds+1, 'Eval Metric'] = scoring_name + ' - std'
    performance_nfold_cv_val.loc[n_folds+2, 'Eval Metric'] = scoring_name + ' - sem'
    performance_nfold_cv_val.loc[n_folds, 'Best Params'] = performance_nfold_cv_val.loc[0, 'Best Params']
    performance_nfold_cv_val.loc[n_folds + 1, 'Best Params'] = performance_nfold_cv_val.loc[0, 'Best Params']
    performance_nfold_cv_val.loc[n_folds + 2, 'Best Params'] = performance_nfold_cv_val.loc[0, 'Best Params']
    # Test
    performance_nfold_cv_test.loc[n_folds, 'Eval Metric'] = scoring_name + ' - mean'
    performance_nfold_cv_test.loc[n_folds+1, 'Eval Metric'] = scoring_name + ' - std'
    performance_nfold_cv_test.loc[n_folds+2, 'Eval Metric'] = scoring_name + ' - sem'
    performance_nfold_cv_test.loc[n_folds, 'Best Params'] = performance_nfold_cv_test.loc[0, 'Best Params']
    performance_nfold_cv_test.loc[n_folds + 1, 'Best Params'] = performance_nfold_cv_test.loc[0, 'Best Params']
    performance_nfold_cv_test.loc[n_folds + 2, 'Best Params'] = performance_nfold_cv_test.loc[0, 'Best Params']

    # Return the results
    return performance_nfold_cv_val, performance_nfold_cv_test


# Create a function that receives a dataframe with models and performs 9-fold cross validation on each model and
# returns the mean, std, sem for every performance metric
def multi_model_nfold_xgb_cv(performance_dataframe, data_train, X_test, y_test, n_folds=9):
    # Get the model ids
    model_ids = performance_dataframe.index.values

    # Initialize the dataframes to store the mean, std and sem of each model for every performance metric
    # Mean
    mean_performance_nfold_cv_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                  'Val - Average Precision', 'Val - Precision',
                                                                  'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                                  'Val - MCC',
                                                                  'Val - AUC', 'Best Params', 'Best Round'],
                                                         index=model_ids)
    mean_performance_nfold_cv_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=model_ids)
    # Std
    std_performance_nfold_cv_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                  'Val - Average Precision', 'Val - Precision',
                                                                  'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                                  'Val - MCC',
                                                                  'Val - AUC', 'Best Params', 'Best Round'],
                                                         index=model_ids)
    std_performance_nfold_cv_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=model_ids)
    # SEM
    sem_performance_nfold_cv_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                                  'Val - Average Precision', 'Val - Precision',
                                                                  'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                                  'Val - MCC',
                                                                  'Val - AUC', 'Best Params', 'Best Round'],
                                                         index=model_ids)
    sem_performance_nfold_cv_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                               'Test - Average Precision', 'Test - Precision',
                                                               'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                               'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                      index=model_ids)

    # Initialize params
    params = {'max_depth': 6,
              'min_child_weight': 5,
              'eta': 0.05,
              'subsample': 1,
              'colsample_bytree': 0.5,
              'scale_pos_weight': 2,
              'objective': 'binary:logistic',
              'seed': 42,
              }

    # Define all scoring functions and function names
    scoring_functions = ['error', 'logloss', 'auc', 'aucpr', 'map', xgb_f1, xgb_balanced_accuracy,
                         xgb_precision, xgb_recall, xgb_mcc]
    scoring_names = ['error', 'logloss', 'auc', 'aucpr', 'map', 'f1-score', 'balanced accuracy', 'precision',
                     'recall', 'mcc']

    # For every model, perform 9-fold cv
    for i in model_ids:
        # Get model params
        best_params = performance_dataframe.loc[i, 'Best Params']
        best_params = best_params.replace('[', '')
        best_params = best_params.replace(']', '')
        best_params = best_params.split(',')
        params['max_depth'] = int(best_params[0])
        params['min_child_weight'] = int(best_params[1])
        params['eta'] = float(best_params[2])
        params['subsample'] = float(best_params[3])
        params['colsample_bytree'] = float(best_params[4])
        params['scale_pos_weight'] = float(best_params[5])
        # Get evaluation metric
        scoring_name = performance_dataframe.loc[i, 'Eval Metric']
        # Get scoring_function
        scoring_function = scoring_functions[np.where([j == scoring_name for j in scoring_names])[0][0]]

        # Perform the 9-fold cv
        performance_nfold_cv_val, performance_nfold_cv_test = \
            nfold_xgb_cv(data_train, X_test, y_test, params, scoring_function, scoring_name)

        # Get the mean, std and sem
        # Mean
        mean_performance_nfold_cv_val.loc[i, :] = performance_nfold_cv_val.loc[n_folds, :]
        mean_performance_nfold_cv_test.loc[i, :] = performance_nfold_cv_test.loc[n_folds, :]
        # Std
        std_performance_nfold_cv_val.loc[i, :] = performance_nfold_cv_val.loc[n_folds+1, :]
        std_performance_nfold_cv_test.loc[i, :] = performance_nfold_cv_test.loc[n_folds+1, :]
        # SEM
        sem_performance_nfold_cv_val.loc[i, :] = performance_nfold_cv_val.loc[n_folds+2, :]
        sem_performance_nfold_cv_test.loc[i, :] = performance_nfold_cv_test.loc[n_folds+2, :]


    # Return the dataframes
    return mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val, \
           std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test


# Create a function that builds a group of XGBoost models where the final prediction is the mean prediction from all
# the models
def build_xgboost_model_group(data, approach, scoring_functions, scoring_names, gridsearch_params, smote_params,
                              boost_rounds, stopping_rounds, folder, th, class_ratio):
    # Split into independent (X) and dependent variables (y)
    X = data.drop(['Reason for discharge as Inpatient'], axis=1).copy()
    y = data['Reason for discharge as Inpatient'].copy()
    y = y.astype(bool)

    # Split into train and val+test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, stratify=y, train_size=0.8)
    # Split val+test into val and test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.5)

    # Create the DMatrix for the test and val datasets
    dval = xgb.DMatrix(data=X_val, label=y_val)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # Find the percentage of positive patients in the dataset
    pos_ratio = y_train.value_counts()[1] / y_train.shape[0]
    # Get the number of patients that belong in the minority class (positive)
    pos_num = y_train.value_counts()[1]
    # Get the number of negative patients to include in every sub-training set
    neg_num = int(pos_num/class_ratio)
    # Merge the X_train and y_train data in X_train
    X_train['Class'] = y_train.copy()
    # Separate the positive from the negative patients
    X_train_pos = X_train[X_train['Class'] == 1].copy()
    X_train_neg = X_train[X_train['Class'] == 0].copy()
    # Initialize a list that will contain the new training datasets with equal class ratios
    train_list = list()
    # Start creating new training datasets with equal class ratios by taking the same positive patients along with
    # random negative patients without replacement from the X_train_neg data. If the X_train_neg data has less
    # patients than the ones required to be sampled, then take the last remaining patients and random sample the
    # rest from the X_train data.
    while X_train_neg.shape[0] != 0:
        if X_train_neg.shape[0] >= neg_num:
            # Get the indices of X_train_neg
            neg_indices = X_train_neg.index.values
            # Get random negative patients
            random.seed(42)
            random_patients = random.sample(neg_indices.tolist(), neg_num)
            # Create the new training dataset
            new_train = X_train_pos.copy()
            new_train = new_train.append(X_train_neg.loc[random_patients, :])
            # Remove the random patients from X_train_neg
            X_train_neg = X_train_neg.drop(labels=random_patients, axis=0)
            # Place the new_train data at the list
            train_list.append(new_train)

        # If the X_train_neg no longer has sufficient number of negative patients, then get the remaining patients
        # and random sample the final ones from the X_train data without repetition.
        else:
            # Create the new training dataset
            new_train = X_train_pos.copy()
            new_train = new_train.append(X_train_neg)
            # Get the number of required patients needed
            required_patients = neg_num - X_train_neg.shape[0]
            # Empty the X_train_neg data
            X_train_neg = pd.DataFrame()
            # Random sample the required patients from the X_train data and place them in the new_train data
            random.seed(42)
            random_patients = random.sample(X_train.index.values.tolist(), required_patients)
            new_train = new_train.append(X_train.loc[random_patients, :])
            # Place the new_train data at the list
            train_list.append(new_train)

    # For every evaluation metric create a model for all combinations
    for i in range(0, len(scoring_functions)):
        if scoring_names[i] in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
            # Initialize params
            params = {'objective': 'binary:logistic',
                      'seed': 42,
                      'tree_method': 'hist',
                      'grow_policy': 'depthwise',
                      'disable_default_eval_metric': True
                      }
        else:
            # Initialize params
            params = {'objective': 'binary:logistic',
                      'seed': 42,
                      'tree_method': 'hist',
                      'grow_policy': 'depthwise',
                      'eval_metric': scoring_functions[i]
                      }

        performance_dataframe_val = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                          'Val - Average Precision', 'Val - Precision',
                                                          'Val - Accuracy', 'Val - Recall', 'Val - F-score',
                                                          'Val - MCC',
                                                          'Val - AUC', 'Best Params', 'Best Round'],
                                                 index=range(0, len(gridsearch_params)))

        performance_dataframe_test = pd.DataFrame(columns=['Eval Metric', 'Val - Best Eval Metric Score',
                                                           'Test - Average Precision', 'Test - Precision',
                                                           'Test - Accuracy', 'Test - Recall', 'Test - F-score',
                                                           'Test - MCC', 'Test - AUC', 'Best Params', 'Best Round'],
                                                  index=range(0, len(gridsearch_params)))

        count = 0
        # Build model using gridsearch
        for max_depth, min_child_weight, eta, subsample, colsample_bytree, scale_pos_weight in gridsearch_params:
            # Update params
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight
            params['eta'] = eta
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample_bytree
            params['scale_pos_weight'] = scale_pos_weight

            # Now for every new_train data, create an XGBoost model and return the performance on the ensemble:
            performance_dataframe_val, performance_dataframe_test, count = \
                build_single_multi_xgboost_model(params, boost_rounds, stopping_rounds, train_list,
                                                            dval, dtest, y_val, y_test,
                                                            scoring_functions[i],
                                                            performance_dataframe_val,
                                                            performance_dataframe_test,
                                                            scoring_names[i], count)

        data_path = folder + r'\Approach ' + str(approach) + r'\Val set'
        filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_val_set' + str(
            int(100 * th)) + '.csv'
        performance_dataframe_val = performance_dataframe_val.sort_values(by='Val - Precision', ascending=False)
        performance_dataframe_val.to_csv(data_path + filename, index=True, header=True)

        data_path = folder + r'\Approach ' + str(approach) + r'\Test set'
        filename = r'\performance_dataframe_' + scoring_names[i] + '_all_models_test_set' + str(
            int(100 * th)) + '.csv'
        performance_dataframe_test = performance_dataframe_test.sort_values(by='Test - Precision', ascending=False)
        performance_dataframe_test.to_csv(data_path + filename, index=True, header=True)


# Create a function that receives a list containing training datasets, creates an xgboost model for every unique dataset
# and performs predictions on the validation and on the test set and returns the mean predictions from all models, thus
# implementing the training of a single xgboost model that is comprised of multiple sub-models.
def build_single_multi_xgboost_model(params, boost_rounds, stopping_rounds, train_list,
                                                            dval, dtest, y_val, y_test,
                                                            scoring_function,
                                                            performance_dataframe_val,
                                                            performance_dataframe_test,
                                                            scoring_function_name, count):
    # Initialize 2 lists: One for the val set and one for the test set, that will contain the predictions of single
    # models for each set and use the mean value to get the model ensemble prediction
    pred_val = list()
    pred_test = list()
    # Initialize the best score and the best rounds that will be calculated as a mean
    model_best_score = 0
    model_best_rounds = 0

    # For every new_train data, create an XGBoost model:
    for j in range(0, len(train_list)):
        X_train = train_list[j]
        y_train = X_train['Class']
        X_train = X_train.drop(columns='Class')

        # Create the DMatrix for the train dataset
        dtrain = xgb.DMatrix(data=X_train, label=y_train)

        # Check whether the scoring function is custom or not
        if scoring_function_name in ['f1-score', 'balanced accuracy', 'precision', 'recall', 'mcc']:
            clf_xgb = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=boost_rounds,
                early_stopping_rounds=stopping_rounds,
                evals=[(dval, "Val")],
                feval=scoring_function
            )
        else:
            clf_xgb = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=boost_rounds,
                early_stopping_rounds=stopping_rounds,
                evals=[(dval, "Val")]
            )

        # Get best score
        model_best_score = model_best_score + abs(clf_xgb.best_score)
        # Get best rounds
        model_best_rounds = model_best_rounds + clf_xgb.best_iteration + 1

        # Make prediction on val set and store it
        y_pred_val = clf_xgb.predict(dval, ntree_limit=clf_xgb.best_iteration + 1)
        pred_val.append(y_pred_val)

        # Make prediction on test set and store it
        y_pred_test = clf_xgb.predict(dtest, ntree_limit=clf_xgb.best_iteration + 1)
        pred_test.append(y_pred_test)

    # Get the average predictions
    for i in range(0, len(pred_val)):
        if i == 0:
            y_pred_val_avg = pred_val[i]
            y_pred_test_avg = pred_test[i]
        else:
            y_pred_val_avg = y_pred_val_avg + pred_val[i]
            y_pred_test_avg = y_pred_test_avg + pred_test[i]

    y_pred_val_avg = y_pred_val_avg/len(pred_val)
    y_pred_test_avg = y_pred_test_avg/len(pred_val)

    # Get best params
    best_params = [params['max_depth'], params['min_child_weight'], params['eta'], params['subsample'],
                   params['colsample_bytree'], params['scale_pos_weight']]
    # Get average model score and average rounds
    model_best_score = model_best_score/len(pred_val)
    model_best_rounds = model_best_rounds/len(pred_val)

    # Now calculate the performance metrics using the model ensemble on the val set
    # Average precision = AP = sum_n((recall_n - recall_n-1)/Pn)
    ap_score = average_precision_score(y_val.values, y_pred_val_avg)
    # Precision score
    check2 = y_pred_val_avg.round()
    prec_score = precision_score(y_val.values, check2)
    # Recall score
    rec_score = recall_score(y_val.values, check2)
    # Accuracy score
    ac_score = accuracy_score(y_val.values, check2)
    # F-score
    f_score = 2 * prec_score * rec_score / (prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_val.values, check2)
    # AUC
    fpr, tpr, _ = roc_curve(y_val.values, y_pred_val_avg)
    roc_auc = auc(fpr, tpr)

    # Store the results in the performance dataframe
    performance_dataframe_val.loc[count, 'Eval Metric'] = scoring_function_name
    performance_dataframe_val.loc[count, 'Val - Best Eval Metric Score'] = model_best_score
    performance_dataframe_val.loc[count, 'Val - Average Precision'] = ap_score
    performance_dataframe_val.loc[count, 'Val - Precision'] = prec_score
    performance_dataframe_val.loc[count, 'Val - Accuracy'] = ac_score
    performance_dataframe_val.loc[count, 'Val - Recall'] = rec_score
    performance_dataframe_val.loc[count, 'Val - F-score'] = f_score
    performance_dataframe_val.loc[count, 'Val - MCC'] = mcc
    performance_dataframe_val.loc[count, 'Val - AUC'] = roc_auc
    performance_dataframe_val.loc[count, 'Best Params'] = best_params
    performance_dataframe_val.loc[count, 'Best Round'] = model_best_rounds

    # Now calculate the performance metrics using the model ensemble on the test set
    # Average precision = AP = sum_n((recall_n - recall_n-1)/Pn)
    ap_score = average_precision_score(y_test.values, y_pred_test_avg)
    # Precision score
    check2 = y_pred_test_avg.round()
    prec_score = precision_score(y_test.values, check2)
    # Recall score
    rec_score = recall_score(y_test.values, check2)
    # Accuracy score
    ac_score = accuracy_score(y_test.values, check2)
    # F-score
    f_score = 2 * prec_score * rec_score / (prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_test.values, check2)
    # AUC
    fpr, tpr, _ = roc_curve(y_test.values, y_pred_test_avg)
    roc_auc = auc(fpr, tpr)

    performance_dataframe_test.loc[count, 'Eval Metric'] = scoring_function_name
    performance_dataframe_test.loc[count, 'Val - Best Eval Metric Score'] = model_best_score
    performance_dataframe_test.loc[count, 'Test - Average Precision'] = ap_score
    performance_dataframe_test.loc[count, 'Test - Precision'] = prec_score
    performance_dataframe_test.loc[count, 'Test - Accuracy'] = ac_score
    performance_dataframe_test.loc[count, 'Test - Recall'] = rec_score
    performance_dataframe_test.loc[count, 'Test - F-score'] = f_score
    performance_dataframe_test.loc[count, 'Test - MCC'] = mcc
    performance_dataframe_test.loc[count, 'Test - AUC'] = roc_auc
    performance_dataframe_test.loc[count, 'Best Params'] = best_params
    performance_dataframe_test.loc[count, 'Best Round'] = model_best_rounds

    count = count + 1

    # Return dataframes and counter
    return performance_dataframe_val, performance_dataframe_test, count



