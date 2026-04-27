# Data loading and splitting utilities 

import pandas as pd 
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv('data/users_behavior.csv')

def split_data(df, target_col='is_ultra', test_size=0.4, valid_size=0.5, random_state=12345):
    # Split data into train (60%), validation (20%), and test(20%) sets
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 60% train and 40% rest
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Split the rest equally into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=valid_size, random_state=random_state)


    return X_train, X_valid, X_test, y_train, y_valid, y_test

