# This script is devoted to viewing the distributions of the data we will be using for building our models

# Modules
import pandas as pd
import numpy as np
from importlib import reload
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun


# 1) Load pre-processed datasets
merged_data_no_td, merged_data_td, merged_data2 = datafun.load_merged_after_comorbidities(2)

for datasets in [0, 1]:
# 2) Choose dataset
    if datasets == 0:
        data = merged_data_no_td.copy()
        folder = r'H:\Documents\Coding\Projects\Thesis\Plots\Mortality Risk\Merged data no td'

        # Use missing value thresholds: 0.8, 0.85, 0.9
        missing_values_th = [0.8, 0.85, 0.9]

    elif datasets == 1:
        data = merged_data2.copy()
        folder = r'H:\Documents\Coding\Projects\Thesis\Plots\Mortality Risk\Merged data2'

        # Use missing value thresholds: 0.9, 0.99
        missing_values_th = [0.9, 0.99]

    # Replace certain unwanted string characters
    data.columns = data.columns.str.replace(',', '')
    data.columns = data.columns.str.replace('[', '(')
    data.columns = data.columns.str.replace(']', ')')

    # Keep only adult people
    data = data[data['Age'] >= 18]

    for th in missing_values_th:
        # Data path
        data_path = folder + r'\Data ' + str(int(100 * th))
        # Keep only patients that have at least th% of their values filled
        data, rows_to_drop, row_nan_num = procfun.drop_rows_with_missing_values(data, th)

        # 2) Split the data into inputs and outputs, the only output being the death or not
        y_no_td = data['Reason for discharge as Inpatient'].copy()
        x_no_td = data[data.columns[~data.columns.isin(['ID', 'Constant record date', 'Sex',
                                                        'COVID diagnosis during admission',
                                                        'Date of Admission as Inpatient',
                                                        'Reason for discharge as Inpatient'])]].copy()


        # 3) Find the indices of the positive and negative cases for every dataset.
        # merged_data_no_td
        indices_dead_no_td = y_no_td[y_no_td == 1].index.values
        indices_alive_no_td = y_no_td[y_no_td == 0].index.values

        # 4) Create a histogram for every value, in order to see how related each input value x is with the output value y.
        # merged_data_no_td
        for i in range(0, x_no_td.shape[1]):
            col_name = x_no_td.columns.values[i]
            positive_values = x_no_td.loc[indices_dead_no_td, col_name].values
            negative_values = x_no_td.loc[indices_alive_no_td, col_name].values
            plotfun.plot_hist_input_distribution(positive_values, negative_values, col_name, i, data_path)
