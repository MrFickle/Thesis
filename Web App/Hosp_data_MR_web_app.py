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

# 3) Homepage
@app.route('/')
def home():
    return render_template('index.html')

# 4) Render results on HTML GUI
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve values from form
    init_features = np.array([x for x in request.form.values()])
    # Get the feature names
    feature_names = clf_xgb.feature_names
    # Get the feature types
    feature_types = clf_xgb.feature_types
    # Pass them into a pandas dataframe
    final_features = pd.DataFrame(data=init_features.reshape(1, len(init_features)), columns=feature_names)
    # Change the dtypes of the features to the ones corresponding to the feature types in the trained model
    for i in range(0, final_features.shape[1]):
        if feature_types[i] == 'int':
            final_features[feature_names[i]] = final_features[feature_names[i]].astype(int)
        elif feature_types[i] == 'float':
            final_features[feature_names[i]] = final_features[feature_names[i]].astype(float)

    # Convert pandas into a Dmatrix
    final_features = xgb.DMatrix(data=final_features, label=pd.Series([1]))
    # Find the prediction probability
    y_prob = clf_xgb.predict(final_features, ntree_limit=clf_xgb.best_iteration + 1)
    # Convert it to percentage
    y_perc = np.round(100*y_prob).astype(int)

    # Return the prediction to the user
    return render_template('index.html', prediction_text='Mortality Risk: {}'.format(y_perc) + ' %.')

if __name__ == "__main__":
    app.run(debug=True)

