#Full training pipeline: load data, tune models, save the best model and evaluate.return

import sys
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

from src.data import load_data, split_data
from src.train import tune_decision_tree, tune_random_forest, evaluate_logistic_regression, train_and_save_best_Random_forest

def main():
    #1. Load and split the data
    df = load_data('data/user_behavior.csv')
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    #2. Evaluate different models on the validation set for the record
    dt_depth, dt_acc = tune_decision_tree(X_train, y_train, X_valid, y_valid)
    rf_params, rf_acc = tune_random_forest(X_train, y_train, X_valid, y_valid)
    lr_acc = evaluate_logistic_regression(X_train, y_train, X_valid, y_valid)

    print(f"Decision tree - best depth: {dt_depth}, val acc: {dt_acc:.4f}")
    print(f"Random forest - {rf_params}, val_acc: {rf_acc:.4f}")
    print(f"Logistic regression - val acc: {lr_acc:.4f}")

    #3. Train the best model (RANDOM FOREST) and save it
    model, best_params, best_val_acc = train_and_save_best_Random_forest(X_train, y_train, X_valid, y_valid)

    #4. Final evaluation on the test set
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    if test_acc >= 0.75:
        print(f"\nModel meets the required accuracy >= 0.75.")
    else:
        print(f"\nModel does not meet the threshold.")

if __name__ == '__main__':
    main()
