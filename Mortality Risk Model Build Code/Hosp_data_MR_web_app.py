import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import xgboost as xgb


# 1) Initialize the Flask app
app = Flask(__name__)

# 2) Load the trained model
data_path = r'H:\Documents\Coding\Projects\Thesis\Web App\Mortality Risk\Implementation 2\Merged data no td'
filename = r'\MR_model_no_td.pkl'
clf_xgb = pickle.load(open(data_path + filename, 'rb'))

# 3) Load column names
filename = r'\column_names.pkl'
column_names = pickle.load(open(data_path + filename, 'rb'))

# 4) Homepage
@app.route('/')
def home():
    return render_template('index.html')

# 5) Render results on HTML GUI
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve values from form
    init_features = [x for x in request.form.values()]
    # Pass them into a pandas dataframe
    final_features = pd.Series(data=init_features, index=column_names)
    # Convert pandas into a Dmatrix
    final_features = xgb.DMatrix(final_features)
    # Find the prediction probability
    y_prob = clf_xgb.predict(final_features, ntree_limit=clf_xgb.best_iteration + 1)
    # Convert it to percentage
    y_perc = np.round(100*y_prob)


    # Return the prediction to the user
    return render_template('index.html', prediction_text='Mortality Risk: {} %'.format(y_perc))

if __name__ == "__main__":
    app.run(debug=True)

