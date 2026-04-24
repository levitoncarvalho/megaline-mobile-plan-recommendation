# Model loading and prediction utilities

import joblib
import pandas as pd


def load_model(model_path='models/random_forest_model.joblib'):
    #Load the saved random forest model
    return joblib.load(model_path)

def make_prediction(model, features):
    #predict the plan for a singular user returning 0 for smart and 1 for ultra
    if isinstance(features, pd.DataFrame):
        features = features.values
    return model.predict(features)[0]
    