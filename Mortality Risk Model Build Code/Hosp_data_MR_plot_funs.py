"""
This script contains functions that are used only for plotting data.
"""

# Modules
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import sem
import dc_stat_think as dcst                       # Used for creating an ECDF from a vector
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, average_precision_score, \
    matthews_corrcoef
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns


# Create a function that plots the ECDF of some vector. Besides the vector, it also requires a title for the plot.
def plot_ECDF(vector, data_path):
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    # Time delay color = red
    col1 = '#FF0000'
    # ECDF 0.05 and 0.95 limits color = purple
    col2 = '#C875E3'
    # Font size
    font_size = 29
    # Get ECDF values
    x, y = dcst.ecdf(vector)
    # Get min and max x values as a multiple of 5 to define xtick step
    min_x = min(x) - (10 - np.mod(abs(min(x)), 10))
    max_x = max(x) + (10 - np.mod(max(x), 10))
    step = 10
    x_ticks = np.arange(min_x, max_x + step, step)
    y_ticks = np.arange(0, 1.1, 0.1)

    # Plot the ECDF
    plt.figure(figsize=(14, 10))
    plt.plot(x, y, linestyle='None', marker='o', markersize=4, color=col1, label='ECDF Distribution of Time Delay')
    plt.xlabel('Time Delay Values', fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)
    plt.ylabel('ECDF', fontsize=font_size)
    plt.yticks(y_ticks, fontsize=font_size)
    plt.title('Time delay between the 1st sensor and lab measurements\ndefined as: 1st sensor date - 1st lab date',
              fontsize=font_size, fontweight='bold')
    # Plot horizontal lines at 0.05 and at 0.95
    plt.hlines(0.05, min_x, max_x, linestyle='--', linewidth=2, color=col2, label='0.05 ECDF threshold')
    plt.hlines(0.95, min_x, max_x, linestyle='--', linewidth=2, color=col2, label='0.95 ECDF threshold')
    # Plot vertical lines at -2 and +2 time delay
    plt.vlines(-2, 0, 1, linestyle='-.', linewidth=2, color=col2, label='-2 Time delay')
    plt.vlines(2, 0, 1, linestyle='-.', linewidth=2, color=col2, label='+2 Time delay')
    plt.legend(frameon=False, fontsize=font_size, loc='center right', labelcolor='linecolor')
    plt.tight_layout()

    # Save figure
    plt.savefig(data_path)
    plt.close()


# Create a function that plots the histogram of a specific type of measurement showing the distribution of the values
# for the positive and the negative covid patients.
def plot_hist_input_distribution(positive_values, negative_values, input_name, input_num, data_path):
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    # Font size = 24
    font_size = 26
    # Positive color = green = #2E6918
    positive_col = '#2E6918'
    # Negative color = red = #FF0000
    negative_col = '#FF0000'

    # Title name
    title = 'Distribution of ' + input_name + ' \nfor dead and alive Covid-19 patients'

    # Get the data length for both vectors, so that we use weights due to different lengths
    data_len_pos = len(positive_values)
    data_len_neg = len(negative_values)

    # Max and min data length
    max_data_length = max([data_len_pos, data_len_neg])
    min_data_length = min([data_len_pos, data_len_neg])

    # Bins
    min_value = np.nanmin(np.concatenate([positive_values, negative_values]))
    max_value = np.nanmax(np.concatenate([positive_values, negative_values]))
    binwidth = (max_value - min_value) / 20
    bin_edges = np.arange(min_value, max_value + binwidth, binwidth)

    # Create the histogram
    plt.figure(figsize=(12, 8))
    y_max = list()
    x_max = list()
    x_min = list()
    legend_handles = list()

    # Positive Covid
    pos_weight = np.ones(data_len_pos) * max_data_length / data_len_pos
    y_vals, x_vals, e = plt.hist(positive_values, bins=bin_edges,
                                 color=positive_col, alpha=0.5,
                                 weights=pos_weight)
    temp = round(max(y_vals) / max_data_length + 0.01, 2)
    y_max.append(temp)
    x_max.append(max(x_vals))
    x_min.append(min(x_vals))
    legend_handles.append(mlines.Line2D([], [], color=positive_col, label='Dead Patients', linestyle='None'))

    # Negative Covid
    neg_weight = np.ones(data_len_neg) * max_data_length / data_len_neg
    y_vals, x_vals, e = plt.hist(negative_values, bins=bin_edges,
                                 color=negative_col, alpha=0.5,
                                 weights=neg_weight)
    temp = round(max(y_vals) / max_data_length + 0.01, 2)
    y_max.append(temp)
    x_max.append(max(x_vals))
    x_min.append(min(x_vals))
    legend_handles.append(mlines.Line2D([], [], color=negative_col, label='Alive Patients', linestyle='None'))

    # Labels
    y_abs_max = 1*max_data_length
    y_tick_interval = 0.1*max_data_length
    x_abs_max = max(x_max)
    x_abs_min = min(x_min)
    x_tick_interval = round((x_abs_max - x_abs_min)/10, 2)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=max_data_length))
    if input_name in ['Cardiac dysrhythmias', 'Chronic Kidney Disease', 'Coronary atherosclerosis', 'Diabetes', 'Severity', 'Sex']:
        plt.xticks([0, 1], fontsize=font_size)
    elif input_name in ['Age', 'Mean Corpuscular Hemoglobin (pg)', 'Aspartate Aminotransferase (U/L)',
                        'Sodium (mmol/L)', 'Prothrombin Time (s)', 'Platelet Count (10^3/μL)',
                        'C-Reactive Protein (mg/L)', 'Blood Glucose (mg/dL)', 'Maximum blood pressure value',
                        'Minimum blood pressure value', 'Heart rate value', 'Alanine Aminotransferase (U/L)',
                        'Oxygen saturation value']:
        plt.xticks(np.arange(int(round(x_abs_min)), int(round(x_abs_max)) + round(x_tick_interval), round(x_tick_interval)),
                   fontsize=font_size)
    elif input_name in ['Creatinine (mg/dL)']:
        plt.xticks(np.arange(round(x_abs_min, 2), round(x_abs_max, 2) + x_tick_interval, x_tick_interval),
                   fontsize=font_size)
    elif input_name in ['Potasium (mmol/L)', 'Leukocytes (10^3/μL)',
                        'Hemoglobin (g/dL)', 'Temperature value']:
        plt.xticks(np.arange(round(x_abs_min, 1), round(x_abs_max, 1) + round(x_tick_interval, 1), round(x_tick_interval, 1)),
                   fontsize=font_size)
    plt.xlabel(input_name, fontsize=font_size)
    plt.ylabel('Percentage of patients (%)', fontsize=font_size)
    plt.yticks(np.arange(0, np.round(y_abs_max) + y_tick_interval, y_tick_interval), fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.legend(handles=[j for j in legend_handles], frameon=False, fontsize=font_size, loc='upper right',
               labelcolor='linecolor')
    plt.tight_layout()

    # Store figure
    filename = r'\hist_dist_of_input' + str(input_num) + '.png'
    plt.savefig(data_path + filename)
    plt.close()


# Create a function that receives a model and plots the confusion matrix, the classification tree, the ROC, the feature
# importance and some evaluation metrics in a barplot.
def plot_model_results(clf_xgb, X_test, y_test, title, data_path_scores, data_path_roc):
    # Create confusion matrix
    #plot_confusion_matrix(clf_xgb, X_test, y_test, values_format='d', display_labels=['Alive', 'Dead'])
    #plt.show()

    '''
    # Plot tree
    bst = clf_xgb.get_booster()
    for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
        print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))
    node_params = {'shape': 'box', 'style': 'filled, rounded', 'fillcolor': '#78cbe'}
    leaf_params = {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}

    xgb.plot_tree(clf_xgb, num_trees=0, size="10,10", condition_node_params=node_params, leaf_node_params=leaf_params)
    fig = plt.gcf()
    fig.set_size_inches(40, 30)
    '''

    # Make prediction on test set
    y_pred = clf_xgb.predict(X_test, ntree_limit=clf_xgb.best_iteration + 1)

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
    f_score = 2*prec_score*rec_score/(prec_score + rec_score)
    # Mathews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_test.values, check2)

    # Plot the above 4 metrics in a barplot
    # Scores
    scores = [ap_score, prec_score, rec_score, ac_score, f_score, mcc]
    # Labels
    labels = ['Average \nPrecision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'Matthews \nCorrCoef']
    # Red, Green, Blue, Purple, Wine, Mango Tango (Bronze like)
    colors = ['#FF0000', '#2E6918', '#0F3CEE', '#C875E3', '#7D3232', '#FF8C4C']
    # Font size
    font_size = 19
    # Label and bar locations
    x = np.arange(1, 3 * len(labels)+1, 3)
    bar_width = 0.6

    # Turn off interactive mode
    plt.ioff()
    # Make the figure
    plt.figure(figsize=(12, 10))
    for i in range(0, len(labels)):
        plt.bar(x=x[i], height=scores[i], width=bar_width, align='center', color=colors[i])
    # Plot the auc score of the covidanalytics.io
    #plt.hlines(y=0.83, xmin=0, xmax=max(x), linestyle='--', label='Covidanalytics.io AUC score',
    #           color=colors[0], alpha=0.5, linewidth=1)
    # Plot max value
    plt.hlines(y=1, xmin=min(x)-0.2*3, xmax=max(x)+0.2*3, linestyle='--', label='Max Value',
               color='black', linewidth=1)
    plt.ylabel('Score value', fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.xticks(x, labels=labels, fontsize=font_size)
    plt.margins(0.2, 0.2)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.legend(frameon=False, fontsize=font_size, loc='upper right', labelcolor='linecolor')
    plt.tight_layout()
    plt.savefig(data_path_scores)
    plt.close()

    # Compute micro-average ROC curve and ROC area and plot the curve
    #y_prob = clf_xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)
    # Turn off interactive mode
    plt.ioff()
    # Make the figure
    plt.figure(figsize=(12, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.legend(loc="upper left", frameon=False, fontsize=font_size)
    plt.tight_layout()
    plt.savefig(data_path_roc)
    plt.close()

    # Print scores
    print('Average Precision:', scores[0])
    print('Precision:', scores[1])
    print('Recall:', scores[2])
    print('Accuracy:', scores[3])
    print('F-score:', scores[4])
    print('Matthews CorrCoef:', scores[5])

    '''
    
    # Plot feature importance
    # plot
    #plt.bar(range(len(clf_xgb.feature_importances_)), clf_xgb.feature_importances_)
    #plt.show()

    # Print feature importance
    xgb.plot_importance(clf_xgb, max_num_features=20)
    plt.show()
    '''


# Create a function that receives an xgboost model and plots the importance of the features in a cdf using an ecdf
# fit and marks the zones per 10% percentile.
def plot_feature_importance_ecdf(clf_xgb, model_id, data_path):
    # Plot params
    font_size = 19
    title = 'Cumulative distribution of feature importance values in model ' + str(model_id)

    # Get the feature importance dict
    feature_imp_dict = clf_xgb.get_fscore()
    # Get the features and their scores
    features = list()
    features_scores = list()
    for i in feature_imp_dict.keys():
        features.append(i)
        features_scores.append(feature_imp_dict[i])

    # Get the min and max values and the step to plot
    min_imp = np.min(features_scores)
    max_imp = np.max(features_scores)
    step_imp = round((max_imp - min_imp) / 10)

    # Define the perc thresholds to plot
    perc_th = [0.9, 0.8, 0.7, 0.6, 0.5]

    # Get the ECDF
    ecdf = ECDF(features_scores, 'right')
    
    # Get the x and y values of the ecdf and set the first x value to 0
    ecdf_x = ecdf.x
    ecdf_x[0] = 0
    ecdf_y = ecdf.y

    # Make the cdf plot
    plt.figure(figsize=(12, 10))
    plt.plot(ecdf_x, ecdf_y, color='#FF0000', linewidth=2, label='ECDF of Feature Importance')
    # Plot lines on the percentile thresholds defined
    for i in perc_th:
        # Find number of features that are above the threshold
        feat_num = len(np.where(ecdf_y >= i)[0])
        # Find the accumulative contribution of the features above the threshold
        feat_cont = round(100*np.sum(ecdf_x[np.where(ecdf_y >= i)[0]])/np.sum(ecdf_x), 1)
        plt.hlines(i, min_imp, max_imp, colors='#C875E3', linestyles='dashed')
        plt.plot(np.min(ecdf_x[np.where(ecdf_y >= i)[0]]), np.min(ecdf_y[np.where(ecdf_y >= i)[0]]), color='#C875E3', linestyle='None', marker='o',
                 markersize=10, label=str(feat_num) + ' features w/ total contribution = ' + str(feat_cont) + '%')
    plt.xlabel('Feature importance value', fontsize=font_size)
    plt.xticks(np.arange(min_imp, max_imp+step_imp, step_imp), fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.ylabel('ECDF', fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.legend(loc="lower right", frameon=False, fontsize=font_size, labelcolor='linecolor')
    plt.tight_layout()
    plt.savefig(data_path + r'\feature_importance_ecdf_model' + str(model_id) + '.png')
    plt.close()


# Create a function that receives a model and plots the performance metrics values.
def plot_model_results2(clf_xgb, dtest, y_test, model_id, data_path):
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

    # Plot the above 4 metrics in a barplot
    # Scores
    scores = [ap_score, prec_score, rec_score, ac_score, f_score, mcc]
    # Labels
    labels = ['Average \nPrecision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'Matthews \nCorrCoef']
    # Red, Green, Blue, Purple, Wine, Mango Tango (Bronze like)
    colors = ['#FF0000', '#2E6918', '#0F3CEE', '#C875E3', '#7D3232', '#FF8C4C']
    # Font size
    font_size = 24
    # Label and bar locations
    x = np.arange(1, 3 * len(labels) + 1, 3)
    bar_width = 0.6

    # Turn off interactive mode
    plt.ioff()
    # Make the figure
    plt.figure(figsize=(18, 12))
    for i in range(0, len(labels)):
        plt.bar(x=x[i], height=scores[i], width=bar_width, align='center', color=colors[i])
    # Plot the auc score of the covidanalytics.io
    # plt.hlines(y=0.83, xmin=0, xmax=max(x), linestyle='--', label='Covidanalytics.io AUC score',
    #           color=colors[0], alpha=0.5, linewidth=1)
    # Plot max value
    plt.hlines(y=1, xmin=min(x) - 0.2 * 3, xmax=max(x) + 0.2 * 3, linestyle='--', label='Max Value',
               color='black', linewidth=1)
    plt.ylabel('Score value', fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.xticks(x, labels=labels, fontsize=font_size)
    plt.margins(0.2, 0.2)
    plt.title('Score of performance metrics on model ' + str(model_id), fontsize=font_size, fontweight='bold')
    plt.legend(frameon=False, fontsize=font_size, loc='upper right', labelcolor='linecolor')
    plt.tight_layout()
    plt.savefig(data_path + r'\performance_metric_scores_model' + str(model_id) + '.png')
    plt.close()


# Create a function that plots the ROC curve of a specific model
def plot_roc_curve(clf_xgb, dtest, y_test, model_id, data_path):
    # Make prediction on test set
    y_pred = clf_xgb.predict(dtest, ntree_limit=clf_xgb.best_iteration + 1)

    # Compute micro-average ROC curve and ROC area and plot the curve
    # y_prob = clf_xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, th = roc_curve(y_test.values, y_pred)
    roc_auc = auc(fpr, tpr)
    # Turn off interactive mode
    plt.ioff()
    # Make the figure
    font_size = 30
    plt.figure(figsize=(12, 12))
    lw = 2
    plt.plot(fpr, tpr, color='#fae100',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title('ROC curve of model ' + str(model_id), fontsize=font_size, fontweight='bold')
    plt.legend(loc="upper left", frameon=False, fontsize=font_size)
    plt.tight_layout()
    plt.savefig(data_path + r'\roc_curve_model' + str(model_id) + '.png')
    plt.close()


# Create a function that receives the evaluation metric values during training for the training and validation sets,
# along with the name of the evaluation metric and plots the learning curve of the model and saves it to the desired
# data path.
# Inputs:
# 1) eval_results: Dictionary in the format dict[eval_set_key][eval_metric_key] = eval_values
# 2) eval_metric: Name of the evaluation metric
# 3) data_path: folder to store result
# 4) model_id: id model number (default = 0)
# 5) eval_set_names: Names of the evaluation sets (default = ['Train', 'Val]
# Ouputs:
# 1) Learning curve png saved at the data_path defined
def plot_learning_curve_xgb(eval_results, eval_metric, data_path, model_id=0, eval_set_names=['Train', 'Val']):
    if eval_metric == 'f1-score':
        eval_metric = 'f1_score'
    # Get the values of the evaluation results for each evaluation set
    eval_set0_values = eval_results[eval_set_names[0]][eval_metric]
    eval_set0_values = [abs(i) for i in eval_set0_values]
    eval_set1_values = eval_results[eval_set_names[1]][eval_metric]
    eval_set1_values = [abs(i) for i in eval_set1_values]

    # Get the number of iterations as the x axis values
    iterations = np.arange(0, len(eval_set0_values), 1)

    # Plot params
    font_size = 40
    title = 'Learning curve of model ' + str(model_id) + '\nusing the eval metric = ' + eval_metric
    colors = ['#00A5FF', '#ff5e61']

    # Make the plot
    plt.ioff()
    plt.figure(figsize=(12, 12))
    # Eval set0
    plt.plot(iterations, eval_set0_values, color=colors[0], linewidth=2, linestyle='solid', label=eval_set_names[0],
             marker='o', markersize=5)
    plt.plot(iterations, eval_set1_values, color=colors[1], linewidth=2, linestyle='solid', label=eval_set_names[1],
             marker='o', markersize=5)
    plt.xlabel('Iterations', fontsize=font_size)
    if model_id == 4263:
        plt.xticks(np.arange(0, int(10 * np.ceil(len(iterations) / 10)) + 1, 60), fontsize=font_size)
    else:
        plt.xticks(np.arange(0, int(10*np.ceil(len(iterations)/10))+1, 30), fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=font_size)
    plt.ylabel('Eval metric (' + eval_metric + ') value', fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.legend(loc="best", frameon=False, fontsize=font_size, labelcolor='linecolor')
    plt.tight_layout()
    plt.savefig(data_path + r'\learning_curve_model' + str(model_id) + '_' + eval_metric + '.png')
    plt.close()


# Create a function that plots the performance metric values for many models in a column for every model that has
# a precision value >= threshold using a specific evaluation metric.
# Inputs:
# 1) performance_dataframe: A dataframe containing the evaluation metrics used, the model params, the model
# id as the row index and the performance metrics of the model.
# 2) eval_set: The evaluation set loaded ('Val' or 'Test)
# 3) data_path: The data path to store the figure
# 4) th: The threshold used when removing data patients based on missing values
# 5) precision_th: The minimum precision value that the models must have in order to be plotted (default = 0.8)
# 6) mcc_th: The minimum mcc value that the models must have in order to be plotted (default = 0.5)
# 7) rec_th: The minimum recall value that the models must have in order to be plotted (default = 0.4)
# 8) ap_th: The minimum average precision value that the models must have in order to be plotted (default = 0.6)
# 9) auc_th: The minimum auc value that the models must have in order to be plotted (default = 0.85)
# 10) f1_th: The minimum f1-score value that the models must have in order to be plotted (default = 0.5)
# 11) top_models: Number of the top models to keep for the plot (default = 15)
# Outputs:
# 1) Multiple model performance metrics png saved at the data_path defined
def plot_multi_model_performance_metrics(performance_dataframe, eval_set, data_path, th, precision_th=0.8, mcc_th=0.5,
                                         rec_th=0.4, ap_th=0.6, auc_th=0.85, f1_th=0.5, top_models=20):
    # Close all figures
    plt.close()
    # Keep only the models that satisfy the precision_th and the mcc_th
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - Precision'] >= precision_th]
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - MCC'] >= mcc_th]
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - Recall'] >= rec_th]
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - Average Precision'] >= ap_th]
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - AUC'] >= auc_th]
    performance_dataframe = performance_dataframe[performance_dataframe[eval_set + ' - F-score'] >= f1_th]

    # Replace the names of certain evaluation metrics so that they are shorter and fit in the x axis
    performance_dataframe = performance_dataframe.replace('f1-score', 'f1')
    performance_dataframe = performance_dataframe.replace('balanced accuracy', 'ba')
    performance_dataframe = performance_dataframe.replace('precision', 'prec')
    performance_dataframe = performance_dataframe.replace('recall', 'rec')
    performance_dataframe = performance_dataframe.replace('logloss', 'logl')

    # Keep only the top_models based on the precision metric
    performance_dataframe = performance_dataframe.sort_values(by=[eval_set + ' - Precision'], ascending=False)
    performance_dataframe = performance_dataframe.iloc[0:top_models, :]

    # Get the model ids, them being the row indices
    model_ids = performance_dataframe.index.values

    # Plot params
    font_size = 24
    gts = u'\u2265'
    title = 'Performance metrics of multiple models \n on the ' + eval_set + ' set that have precision ' + gts + ' ' + str(precision_th) + \
            ',\naverage precision ' + gts + ' ' + str(ap_th) + ', f1-score ' + gts + ' ' + str(f1_th) + '\nand mcc ' + \
            gts + ' ' + str(mcc_th) + ', recall ' + gts + ' ' + str(rec_th) + ', auc ' + gts + ' ' + str(auc_th)



    # Color dictionary
    color_dict = {}
    color_dict['Average Precision'] = '#FF0000'
    color_dict['Precision'] = '#2E6918'
    color_dict['Recall'] = '#0F3CEE'
    color_dict['Accuracy'] = '#C875E3'
    color_dict['F-score'] = '#a85c32'
    color_dict['MCC'] = '#FF8C4C'
    color_dict['AUC'] = '#fae100'
    # Labels
    labels = ['Average Precision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'MCC', 'AUC']

    # Initialize the figure
    plt.ioff()
    plt.figure(figsize=(18, 6.5))

    # For every model get the performance metric scores and plot them in a scatter plot
    for i in range(0, len(model_ids)):
        # Model id
        model = model_ids[i]
        # Average Precision score
        ap_score = round(performance_dataframe.loc[model, eval_set + ' - Average Precision'], 2)
        plt.scatter(i, ap_score, c=color_dict['Average Precision'], marker='o', s=150)
        # Precision score
        prec_score = round(performance_dataframe.loc[model, eval_set + ' - Precision'], 2)
        plt.scatter(i, prec_score, c=color_dict['Precision'], marker='o', s=150)
        # Recall score
        rec_score = round(performance_dataframe.loc[model, eval_set + ' - Recall'], 2)
        plt.scatter(i, rec_score, c=color_dict['Recall'], marker='o', s=150)
        # Accuracy score
        ac_score = round(performance_dataframe.loc[model, eval_set + ' - Accuracy'], 2)
        plt.scatter(i, ac_score, c=color_dict['Accuracy'], marker='o', s=150)
        # F-score score
        f_score = round(performance_dataframe.loc[model, eval_set + ' - F-score'], 2)
        plt.scatter(i, f_score, c=color_dict['F-score'], marker='o', s=150)
        # MCC score
        mcc = round(performance_dataframe.loc[model, eval_set + ' - MCC'], 2)
        plt.scatter(i, mcc, c=color_dict['MCC'], marker='o', s=150)
        # AUC score
        auc = round(performance_dataframe.loc[model, eval_set + ' - AUC'], 2)
        plt.scatter(i, auc, c=color_dict['AUC'], marker='o', s=150)

    # Labels
    plt.xlabel('Model ID', fontsize=font_size)
    plt.xticks(np.arange(0, len(model_ids)+6, 1), fontsize=16,
               labels=[str(i) + '\n' + performance_dataframe.loc[i, 'Eval Metric'] for i in model_ids] + ['', '', '', '', '', ''])
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.ylabel('Performance metric value', fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    # Create legends
    legends_list = list()
    for i in range(0, len(labels)):
        legends_list.append(mlines.Line2D([], [], linestyle='None', color=color_dict[labels[i]], label=labels[i]))
    plt.legend(handles=[i for i in legends_list], loc="upper right", frameon=False, fontsize=font_size,
               labelcolor='linecolor')

    # Save figure
    plt.savefig(data_path + r'\multi_model_performance_metrics_prec_th' + str(precision_th) + '_mcc_th' + \
                str(mcc_th) + '_rec_th' + str(rec_th) + '_ap_th' + str(ap_th) + '_auc_th' + str(auc_th) + \
                '_f1_th' + str(f1_th) + '_data' + str(int(100 * th)) + '.png')
    plt.close()

    # Return best models
    return model_ids


# Create a function that plots the performance metric values for many models in a column for every model that has
# a precision value >= threshold using a specific evaluation metric.
# Inputs:
# 1) performance_dataframe: A dataframe containing the evaluation metrics used, the model params, the model
# id as the row index and the performance metrics of the model.
# 2) eval_set: The evaluation set loaded ('Val' or 'Test)
# 3) data_path: The data path to store the figure
# 4) th: The threshold used when removing data patients based on missing values
# Outputs:
# 1) Multiple model performance metrics png saved at the data_path defined
def plot_best_models_performance_metrics(performance_dataframe, eval_set, data_path, th):
    # Replace the names of certain evaluation metrics so that they are shorter and fit in the x axis
    performance_dataframe = performance_dataframe.replace('f1-score', 'f1')
    performance_dataframe = performance_dataframe.replace('balanced accuracy', 'ba')
    performance_dataframe = performance_dataframe.replace('precision', 'prec')
    performance_dataframe = performance_dataframe.replace('recall', 'rec')
    performance_dataframe = performance_dataframe.replace('logloss', 'logl')

    # Get the model ids, them being the row indices
    model_ids = performance_dataframe.index.values

    # Plot params
    font_size = 24
    gts = u'\u2265'
    title = 'Performance metrics on the ' + eval_set + ' set of best models'

    # Color dictionary
    color_dict = {}
    color_dict['Average Precision'] = '#FF0000'
    color_dict['Precision'] = '#2E6918'
    color_dict['Recall'] = '#0F3CEE'
    color_dict['Accuracy'] = '#C875E3'
    color_dict['F-score'] = '#a85c32'
    color_dict['MCC'] = '#FF8C4C'
    color_dict['AUC'] = '#fae100'
    # Labels
    labels = ['Average Precision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'MCC', 'AUC']

    # Initialize the figure
    plt.ioff()
    plt.figure(figsize=(18, 6.5))

    # For every model get the performance metric scores and plot them in a scatter plot
    for i in range(0, len(model_ids)):
        # Model id
        model = model_ids[i]
        # Average Precision score
        ap_score = round(performance_dataframe.loc[model, eval_set + ' - Average Precision'], 2)
        plt.scatter(i, ap_score, c=color_dict['Average Precision'], marker='o', s=150)
        # Precision score
        prec_score = round(performance_dataframe.loc[model, eval_set + ' - Precision'], 2)
        plt.scatter(i, prec_score, c=color_dict['Precision'], marker='o', s=150)
        # Recall score
        rec_score = round(performance_dataframe.loc[model, eval_set + ' - Recall'], 2)
        plt.scatter(i, rec_score, c=color_dict['Recall'], marker='o', s=150)
        # Accuracy score
        ac_score = round(performance_dataframe.loc[model, eval_set + ' - Accuracy'], 2)
        plt.scatter(i, ac_score, c=color_dict['Accuracy'], marker='o', s=150)
        # F-score score
        f_score = round(performance_dataframe.loc[model, eval_set + ' - F-score'], 2)
        plt.scatter(i, f_score, c=color_dict['F-score'], marker='o', s=150)
        # MCC score
        mcc = round(performance_dataframe.loc[model, eval_set + ' - MCC'], 2)
        plt.scatter(i, mcc, c=color_dict['MCC'], marker='o', s=150)
        # AUC score
        auc = round(performance_dataframe.loc[model, eval_set + ' - AUC'], 2)
        plt.scatter(i, auc, c=color_dict['AUC'], marker='o', s=150)

    # Labels
    plt.xlabel('Model ID', fontsize=font_size)
    plt.xticks(np.arange(0, len(model_ids)+6, 1), fontsize=16,
               labels=[str(i) + '\n' + performance_dataframe.loc[i, 'Eval Metric'] for i in model_ids] + ['', '', '', '', '', ''])
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
    plt.ylabel('Performance metric value', fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    # Create legends
    legends_list = list()
    for i in range(0, len(labels)):
        legends_list.append(mlines.Line2D([], [], linestyle='None', color=color_dict[labels[i]], label=labels[i]))
    plt.legend(handles=[i for i in legends_list], loc="upper right", frameon=False, fontsize=font_size,
               labelcolor='linecolor')

    # Save figure
    plt.savefig(data_path + r'\best_models_performance_metrics_' + eval_set + '_data' + str(int(100 * th)) + '.png')
    plt.close()


# Create a function that plots the feature importance of a built model in a bar plot
def plot_feature_importance_bar(clf_xgb, model_id, data_path):
    # Plot params
    font_size = 30
    title = 'Feature\nimportance\nin model ' + str(model_id)

    # Get the feature importance dict
    feature_imp_dict = clf_xgb.get_fscore()
    # Get the features and their scores
    features = list()
    features_scores = list()
    for i in feature_imp_dict.keys():
        features.append(i)
        features_scores.append(feature_imp_dict[i])

    # Place the features and their values in a dataframe
    feature_dataframe = pd.DataFrame(data={'Feature Name': features, 'F-Score': features_scores})
    # Sort the features in descending order
    feature_dataframe = feature_dataframe.sort_values(by=['F-Score'], ascending=True)

    # Make a horizontal barplot
    plt.figure(figsize=(12, 12))
    y_loc = np.arange(0, 2*len(features), 2)
    plt.barh(y=y_loc, width=feature_dataframe['F-Score'].values, height=0.8, align='center', color='#00A5FF')
    plt.xlabel('F-Score', fontsize=font_size)
    plt.xticks(np.arange(0, np.max(features_scores)+100, 100), fontsize=font_size)
    plt.yticks(y_loc, labels=feature_dataframe['Feature Name'].values, fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(data_path + r'\features_importance_model' + str(model_id) + '.png')
    plt.close()


# Create a function that plots the confusion matrix of a model
def plot_confusion_matrix(clf_xgb, dtest, y_test, model_id, data_path):
    # Make prediction on test set
    y_pred = clf_xgb.predict(dtest, ntree_limit=clf_xgb.best_iteration + 1)
    # Precision score
    check2 = y_pred.round()

    # Font size
    font_size = 40

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=check2)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(conf_matrix, cmap=plt.cm.plasma, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', fontsize=font_size)

    plt.xlabel('Predictions', fontsize=font_size)
    plt.xticks([0, 1], fontsize=font_size, labels=['Alive', 'Dead'])
    plt.ylabel('Actuals', fontsize=font_size)
    plt.yticks([0, 1], fontsize=font_size, labels=['Alive', 'Dead'])
    plt.title('Confusion Matrix\nof Model ' + str(model_id), fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(data_path + r'\confusion_matrix_model' + str(model_id) + '.png')
    plt.close(fig)


# Create function that receives the mean, std, sem values of the performance metrics of models after using 9-fold cv
# and plots the top 20 models based on 2 specific metrics:
# Metric 1: mean, sem, mean/sem
# Metric 2: name of performance metric
def plot_best_cv_models(mean_performance_nfold_cv_val, mean_performance_nfold_cv_test, sem_performance_nfold_cv_val,
                        sem_performance_nfold_cv_test, metric1, metric2, data_path, th, eval_set, top_models=20):
    # Define whether the top models picked need to have the highest or the lowest metric and also choose the relative
    # dataframe to use
    if metric1 == 'mean':
        ascending = False
        data_val = mean_performance_nfold_cv_val
        data_test = mean_performance_nfold_cv_test
    elif metric1 == 'sem':
        ascending = True
        data_val = sem_performance_nfold_cv_val
        data_test = sem_performance_nfold_cv_test
    elif metric1 == 'mean_sem':
        ascending = False
        data_val = mean_performance_nfold_cv_val.copy()
        data_test = mean_performance_nfold_cv_test.copy()
        data_val.iloc[:, 2:9] = data_val.iloc[:, 2:9] / sem_performance_nfold_cv_val.iloc[:, 2:9]
        data_test.iloc[:, 2:9] = data_test.iloc[:, 2:9] / sem_performance_nfold_cv_test.iloc[:, 2:9]

    if metric2 in ['Average Precision', 'Precision', 'AUC', 'Recall', 'Accuracy', 'F-score', 'MCC']:
        if eval_set == 'Val':
            # Sort models based on metric
            data_val = data_val.sort_values(by=eval_set + ' - ' + metric2, ascending=ascending)
            # Keep top models
            data_val = data_val.iloc[0:top_models, :]
            # Get model ids
            model_ids = data_val.index.values
            # Get the same models for test
            data_test = data_test.loc[model_ids, :]

        elif eval_set == 'Test':
            # Sort models based on metric
            data_test = data_test.sort_values(by=eval_set + ' - ' + metric2, ascending=ascending)
            # Keep top models
            data_test = data_test.iloc[0:top_models, :]
            # Get model ids
            model_ids = data_test.index.values
            # Get the same models for test
            data_val = data_val.loc[model_ids, :]

        # Replace the names of certain evaluation metrics so that they are shorter and fit in the x axis
        # Val
        data_val = data_val.replace('f1-score - mean', 'f1')
        data_val = data_val.replace('balanced accuracy - mean', 'ba')
        data_val = data_val.replace('precision - mean', 'prec')
        data_val = data_val.replace('recall - mean', 'rec')
        data_val = data_val.replace('logloss - mean', 'logl')
        data_val = data_val.replace('aucpr - mean', 'aucpr')
        data_val = data_val.replace('auc - mean', 'auc')
        data_val = data_val.replace('map - mean', 'map')
        data_val = data_val.replace('error - mean', 'error')
        data_val = data_val.replace('mcc - mean', 'mcc')
        
        # Test
        data_test = data_test.replace('f1-score - mean', 'f1')
        data_test = data_test.replace('balanced accuracy - mean', 'ba')
        data_test = data_test.replace('precision - mean', 'prec')
        data_test = data_test.replace('recall - mean', 'rec')
        data_test = data_test.replace('logloss - mean', 'logl')
        data_test = data_test.replace('aucpr - mean', 'aucpr')
        data_test = data_test.replace('auc - mean', 'auc')
        data_test = data_test.replace('map - mean', 'map')
        data_test = data_test.replace('error - mean', 'error')
        data_test = data_test.replace('mcc - mean', 'mcc')


        # Plot params
        font_size = 24
        gts = u'\u2265'
        # Color dictionary
        color_dict = {}
        color_dict['Average Precision'] = '#FF0000'
        color_dict['Precision'] = '#2E6918'
        color_dict['Recall'] = '#0F3CEE'
        color_dict['Accuracy'] = '#C875E3'
        color_dict['F-score'] = '#a85c32'
        color_dict['MCC'] = '#FF8C4C'
        color_dict['AUC'] = '#fae100'
        # Labels
        labels = ['Average Precision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'MCC', 'AUC']



        # Make the plot for the test set
        # Initialize the figure
        plt.ioff()
        plt.figure(figsize=(18, 6.5))
        title = 'Performance metrics on the Test set of best models from 9-fold CV'

        # For every model get the performance metric scores and plot them in a scatter plot
        for i in range(0, len(model_ids)):
            # Model id
            model = model_ids[i]
            # Average Precision score
            ap_score = round(mean_performance_nfold_cv_test.loc[model, 'Test - Average Precision'], 2)
            plt.scatter(i, ap_score, c=color_dict['Average Precision'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = ap_score + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Average Precision']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Average Precision'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = ap_score - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Average Precision']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Average Precision'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Average Precision'], alpha=0.2, linewidth=6)

            # Precision score
            prec_score = round(mean_performance_nfold_cv_test.loc[model, 'Test - Precision'], 2)
            plt.scatter(i, prec_score, c=color_dict['Precision'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = prec_score + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Precision']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Precision'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = prec_score - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Precision']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Precision'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Precision'], alpha=0.2, linewidth=6)

            # Recall score
            rec_score = round(mean_performance_nfold_cv_test.loc[model, 'Test - Recall'], 2)
            plt.scatter(i, rec_score, c=color_dict['Recall'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = rec_score + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Recall']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Recall'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = rec_score - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Recall']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Recall'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Recall'], alpha=0.2, linewidth=6)

            # Accuracy score
            ac_score = round(mean_performance_nfold_cv_test.loc[model, 'Test - Accuracy'], 2)
            plt.scatter(i, ac_score, c=color_dict['Accuracy'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = ac_score + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Accuracy']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Accuracy'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = ac_score - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - Accuracy']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Accuracy'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Accuracy'], alpha=0.2, linewidth=6)

            # F-score score
            f_score = round(mean_performance_nfold_cv_test.loc[model, 'Test - F-score'], 2)
            plt.scatter(i, f_score, c=color_dict['F-score'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = f_score + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - F-score']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['F-score'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = f_score - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - F-score']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['F-score'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['F-score'], alpha=0.2, linewidth=6)

            # MCC score
            mcc = round(mean_performance_nfold_cv_test.loc[model, 'Test - MCC'], 2)
            plt.scatter(i, mcc, c=color_dict['MCC'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = mcc + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - MCC']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['MCC'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = mcc - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - MCC']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['MCC'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['MCC'], alpha=0.2, linewidth=6)

            # AUC score
            auc = round(mean_performance_nfold_cv_test.loc[model, 'Test - AUC'], 2)
            plt.scatter(i, auc, c=color_dict['AUC'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = auc + 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - AUC']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['AUC'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = auc - 1.96*sem_performance_nfold_cv_test.loc[model, 'Test - AUC']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['AUC'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['AUC'], alpha=0.2, linewidth=6)

        # Labels
        plt.xlabel('Model ID', fontsize=font_size)
        plt.xticks(np.arange(0, len(model_ids)+6, 1), fontsize=16,
                   labels=[str(i) + '\n' + data_test.loc[i, 'Eval Metric'] for i in model_ids] + ['', '', '', '', '', ''])
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
        plt.ylabel('Performance metric value', fontsize=font_size)
        plt.title(title, fontsize=font_size, fontweight='bold')
        plt.tight_layout()
        # Create legends
        legends_list = list()
        for i in range(0, len(labels)):
            legends_list.append(mlines.Line2D([], [], linestyle='None', color=color_dict[labels[i]], label=labels[i]))
        plt.legend(handles=[i for i in legends_list], loc="upper right", frameon=False, fontsize=font_size,
                   labelcolor='linecolor')

        # Save figure
        plt.savefig(data_path + r'\best_models_performance_metrics_on_test_set_data' + str(int(100 * th)) +
                    '_' + metric1 + '_' + metric2 + '.png')
        plt.close()

        # Make the plot for the val set
        # Initialize the figure
        plt.ioff()
        plt.figure(figsize=(18, 6.5))
        title = 'Performance metrics on the Val set of best models from 9-fold CV'

        # For every model get the performance metric scores and plot them in a scatter plot
        for i in range(0, len(model_ids)):
            # Model id
            model = model_ids[i]
            # Average Precision score
            ap_score = round(mean_performance_nfold_cv_val.loc[model, 'Val - Average Precision'], 2)
            plt.scatter(i, ap_score, c=color_dict['Average Precision'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = ap_score + 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Average Precision']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Average Precision'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = ap_score - 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Average Precision']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Average Precision'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Average Precision'], alpha=0.2, linewidth=6)

            # Precision score
            prec_score = round(mean_performance_nfold_cv_val.loc[model, 'Val - Precision'], 2)
            plt.scatter(i, prec_score, c=color_dict['Precision'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = prec_score + 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Precision']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Precision'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = prec_score - 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Precision']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Precision'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Precision'], alpha=0.2, linewidth=6)

            # Recall score
            rec_score = round(mean_performance_nfold_cv_val.loc[model, 'Val - Recall'], 2)
            plt.scatter(i, rec_score, c=color_dict['Recall'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = rec_score + 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Recall']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Recall'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = rec_score - 1.96*sem_performance_nfold_cv_val.loc[model, 'Val - Recall']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Recall'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Recall'], alpha=0.2, linewidth=6)

            # Accuracy score
            ac_score = round(mean_performance_nfold_cv_val.loc[model, 'Val - Accuracy'], 2)
            plt.scatter(i, ac_score, c=color_dict['Accuracy'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = ac_score + 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - Accuracy']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['Accuracy'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = ac_score - 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - Accuracy']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['Accuracy'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['Accuracy'], alpha=0.2, linewidth=6)

            # F-score score
            f_score = round(mean_performance_nfold_cv_val.loc[model, 'Val - F-score'], 2)
            plt.scatter(i, f_score, c=color_dict['F-score'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = f_score + 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - F-score']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['F-score'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = f_score - 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - F-score']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['F-score'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['F-score'], alpha=0.2, linewidth=6)

            # MCC score
            mcc = round(mean_performance_nfold_cv_val.loc[model, 'Val - MCC'], 2)
            plt.scatter(i, mcc, c=color_dict['MCC'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = mcc + 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - MCC']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['MCC'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = mcc - 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - MCC']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['MCC'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['MCC'], alpha=0.2, linewidth=6)

            # AUC score
            auc = round(mean_performance_nfold_cv_val.loc[model, 'Val - AUC'], 2)
            plt.scatter(i, auc, c=color_dict['AUC'], marker='o', s=150)
            # Plot the upper 1.96 SEM
            upper_sem = auc + 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - AUC']
            plt.plot(i, upper_sem, linestyle='None', color=color_dict['AUC'], alpha=0.6)
            # Plot the lower 1.96 SEM
            lower_sem = auc - 1.96 * sem_performance_nfold_cv_val.loc[model, 'Val - AUC']
            plt.plot(i, lower_sem, linestyle='None', color=color_dict['AUC'], alpha=0.6)
            # Fill the in betweens
            plt.fill_between([i], [upper_sem], [lower_sem], color=color_dict['AUC'], alpha=0.2, linewidth=6)

        # Labels
        plt.xlabel('Model ID', fontsize=font_size)
        plt.xticks(np.arange(0, len(model_ids) + 6, 1), fontsize=16,
                   labels=[str(i) + '\n' + data_val.loc[i, 'Eval Metric'] for i in model_ids] + ['', '', '', '', '', ''
                                                                                                 ])
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font_size)
        plt.ylabel('Performance metric value', fontsize=font_size)
        plt.title(title, fontsize=font_size, fontweight='bold')
        plt.tight_layout()
        # Create legends
        legends_list = list()
        for i in range(0, len(labels)):
            legends_list.append(mlines.Line2D([], [], linestyle='None', color=color_dict[labels[i]], label=labels[i]))
        plt.legend(handles=[i for i in legends_list], loc="upper right", frameon=False, fontsize=font_size,
                   labelcolor='linecolor')

        # Save figure
        plt.savefig(data_path + r'\best_models_performance_metrics_on_val_set_data' + str(int(100 * th)) +
                    '_' + metric1 + '_' + metric2 + '.png')
        plt.close()

        # Return the model ids
        return model_ids


# Create a function that receives the performance metrics of various models and plots the histogram distribution
def plot_hist_pdf_performance_metrics(performance_metrics, eval_set, title, data_path, th):
    # Plot params
    font_size = 24
    # Color dictionary
    color_dict = {}
    color_dict['Average Precision'] = '#FF0000'
    color_dict['Precision'] = '#2E6918'
    color_dict['Recall'] = '#0F3CEE'
    color_dict['Accuracy'] = '#C875E3'
    color_dict['F-score'] = '#a85c32'
    color_dict['MCC'] = '#FF8C4C'
    color_dict['AUC'] = '#fae100'
    # Labels
    labels = ['Average Precision', 'Precision', 'Recall', 'Accuracy', 'F-score', 'MCC', 'AUC']

    # Make the plot
    plt.ioff()
    plt.figure(figsize=(14, 8))
    for i in range(0, len(labels)):
        if eval_set in ['Val', 'Test']:
            fig = sns.kdeplot(data=performance_metrics[labels[i] + ' - ' + eval_set], fill=True, color=color_dict[labels[i]],
                        cut=0)
        else:
            fig = sns.kdeplot(data=performance_metrics[labels[i]], fill=True, color=color_dict[labels[i]], cut=0)

    # Get x and y lims to define x and y ticks
    xlims = fig.get_xlim()
    xrange = np.ceil(xlims[1]) - np.floor(xlims[0])
    xstep = round(xrange/10, 1)
    xticks = np.arange(np.floor(xlims[0]), np.ceil(xlims[1]), xstep)

    ylims = fig.get_ylim()
    yrange = np.ceil(ylims[1]) - np.floor(ylims[0])
    ystep = round(yrange / 10, 1)
    yticks = np.arange(np.floor(ylims[0]), np.ceil(ylims[1]), ystep)

    plt.xlabel('Performance Metric Values', fontsize=font_size)
    plt.xticks(xticks, fontsize=font_size)
    plt.ylabel('Density', fontsize=font_size)
    plt.yticks(yticks, fontsize=font_size)
    plt.title(title, fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    # Create legends
    legends_list = list()
    for i in range(0, len(labels)):
        legends_list.append(mlines.Line2D([], [], linestyle='None', color=color_dict[labels[i]], label=labels[i]))
    plt.legend(handles=[i for i in legends_list], loc="upper right", frameon=False, fontsize=font_size,
               labelcolor='linecolor')

    # Save figure
    plt.savefig(data_path + r'\performance_metrics_hist_pdf_data' + str(int(100 * th)) + '.png')
    plt.close()



