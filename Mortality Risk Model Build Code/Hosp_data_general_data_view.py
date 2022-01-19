# This script is devoted to viewing the distributions of the data we will be using for building our models

# Modules
import pandas as pd
import numpy as np
from importlib import reload
import Hosp_data_MR_data_funs as datafun
import Hosp_data_MR_proc_funs as procfun
import Hosp_data_MR_plot_funs as plotfun


# 1) Load specific datasets
data_name = 'merged data2'
category = 1
sub_category = 'None'
th = 0.9

if category in [3, 4]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)
    data1 = data[0]
    data2 = data[1]
elif category in [1, 2]:
    data = datafun.load_specific_data(data_name, category, sub_category, th)

# Replace certain unwanted string characters
data.columns = data.columns.str.replace(',', '')
data.columns = data.columns.str.replace('[', '(')
data.columns = data.columns.str.replace(']', ')')

if data_name == 'merged data no td':
    column_names = data.columns.values.tolist()
    loc1 = np.frompyfunc(lambda x: 'Mean Corpuscular Volume (pg)' in x, 1, 1)(data.columns.values)
    loc1 = np.where(loc1)[0][0]
    column_names[loc1] = 'Mean Corpuscular Hemoglobin (pg)'
    data.columns = column_names

# Keep only adult people
data = data[data['Age'] >= 18]

folder = r'H:\Documents\Coding\Projects\Thesis\Plots\Mortality Risk\Merged data2\Presented'

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
