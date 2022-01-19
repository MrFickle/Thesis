# Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun
import Hosp_data_MR_model_funs as modelfun
from importlib import reload


'''
GENERAL STEPS
'''
# 1) Load specific datasets
data_name = 'merged data no td'
category = 2
sub_category = 'None'
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

# 2) Define folders where grid search models are stored and get all models
folder_val = r'H:\Documents\Coding\Projects\Thesis\Models\MR + SR\Merged data no td\Approach 1\All patients\Val Set'
folder_test = r'H:\Documents\Coding\Projects\Thesis\Models\MR + SR\Merged data no td\Approach 1\All patients\Test Set'

performance_dataframe_val, performance_dataframe_test = datafun.load_gridsearch_models(folder_val, folder_test, th)
#a = performance_dataframe_val.describe()
#b = performance_dataframe_test.describe()

# Keep balanced models
#z_score = 1
#model_dif, z_score_dif, models_to_keep = procfun.keep_balanced_models(performance_dataframe_val, performance_dataframe_test, z_score)
#title = 'Distribution of z score performance metric difference between val and test'
#data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Test\Approach 1\Visual Results'
#plotfun.plot_hist_pdf_performance_metrics(z_score_dif, None, title, data_path, th)

# 3) Filter models to keep based on specific performance metric thresholds
eval_set = 'Val'
performance_dataframe_val, performance_dataframe_test = \
    procfun.filter_models_based_on_performance_metrics(performance_dataframe_val, performance_dataframe_test, eval_set,
                                                       precision_th=0.5, mcc_th=0.4, rec_th=0.3, ap_th=0.5, auc_th=0.7,
                                                       f1_th=0.5)
eval_set = 'Test'
performance_dataframe_val, performance_dataframe_test = \
    procfun.filter_models_based_on_performance_metrics(performance_dataframe_val, performance_dataframe_test, eval_set,
                                                       precision_th=0.5, mcc_th=0.4, rec_th=0.3, ap_th=0.5, auc_th=0.7,
                                                       f1_th=0.5)

# 4) For all of the models kept, perform 9-fold cv and return the mean, std, sem of each performance metric in separate
# dataframes
store = False
data_path = r'H:\Documents\Coding\Projects\Thesis\Models\MR + SR\Merged data no td\Approach 1\Diabetes Only\Negative Patients\CV Results'
#data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Mortality Risk\Implementation 2\Merged data no td com\Tuning 3\Approach 1\All patients\CV Results'
if store:
    mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val, \
               std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test = \
        modelfun.multi_model_nfold_xgb_cv(performance_dataframe_val, data_train, X_test, y_test, n_folds=9)

    datafun.store_cv_models(mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val,
                    std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test,
                    performance_dataframe_val, performance_dataframe_test, data_path, th)
else:
    mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, std_performance_nfold_cv_val, \
    std_performance_nfold_cv_test, sem_performance_nfold_cv_val, sem_performance_nfold_cv_test, \
    performance_dataframe_val, performance_dataframe_test = datafun.load_cv_models(data_path, th)


# 5) Plot the performance metrics of the top 20 models based on a metric of choice
model_ids = list()
#for metric1 in ['mean', 'sem', 'mean_sem']:
#    for metric2 in ['MCC', 'Precision']:
metric1 = 'mean'
metric2 = 'MCC'
        data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Test\Method 3\merged data no td severity\Diabetes\Negative Patients'
        #metric1 = 'mean_sem'
        #metric2 = 'MCC'
        eval_set = 'Val'
        temp = plotfun.plot_best_cv_models(mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, sem_performance_nfold_cv_val,
                                sem_performance_nfold_cv_test, metric1, metric2, data_path, th, eval_set, top_models=20)
        model_ids.append(temp)

# 6) Choose which metric1 and metric2 to use to get the best model ids and plot their performance without CV
model_ids = model_ids[0]
#performance_dataframe_val, performance_dataframe_test = datafun.load_gridsearch_models(folder_val, folder_test, th)
performance_dataframe_val = performance_dataframe_val.loc[model_ids, :]
performance_dataframe_test = performance_dataframe_test.loc[model_ids, :]

data_path = r'H:\Documents\Coding\Projects\Thesis\Models\Test\Method 3\merged data no td severity\Diabetes\Negative Patients'
plotfun.plot_best_models_performance_metrics(performance_dataframe_val, 'Val', data_path, th)
plotfun.plot_best_models_performance_metrics(performance_dataframe_test, 'Test', data_path, th)


