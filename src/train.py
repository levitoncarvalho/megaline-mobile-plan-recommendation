# Model training and hyperparameter tuning utilities 

import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def tune_decision_tree(X_train, y_train, X_valid, y_valid):
    # Find the best max_depth for a Decision Tree using the validation set

    best_acc = 0
    best_depth = None
    for depth in range(1, 11):
        model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_valid, model.predict(X_valid))
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
    
    return best_depth, best_acc

def tune_random_forest(X_train, y_train, X_valid, y_valid):
    #Search for the best n_estimators and max_depth for a Random Forest

    best_acc = 0
    best_params = {}
    for n_est in range(10, 51, 10):
        for depth in range(1, 11):
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=12345)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_valid, model.predict(X_valid))
            if acc > best_acc:
                best_acc = acc
                best_params = {'n_estimators': n_est, 'max_depth': depth}
    return best_params, best_acc

def evaluate_logistic_regression(X_train, y_train, X_valid, y_valid):
    # Train a logisitc regression model and return its validation accuracy
    model = LogisticRegression(random_state=12345, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_valid, model.predict(X_valid))
    return acc

def train_and_save_best_Random_forest(X_train, y_train, X_valid, y_valid, model_dir='models'):
    #Find the best Random Forest hyperparameters using the validation set, train a final model and save it.
    best_params, best_acc = tune_random_forest(X_train, y_train, X_valid, y_valid)

    final_model = RandomForestClassifier(
        n_estimators = best_params['n_estimators'],
        max_depth = best_params['max_depth'],
        random_state=12345
    )
    final_model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'random_forest_model.joblib')
    joblib.dump(final_model, model_path)
    print(f"Model saved to: {model_path}")

    return final_model, best_params, best_acc