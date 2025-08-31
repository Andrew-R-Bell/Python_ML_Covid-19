import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.feature_selection import RFECV

#COVID-19 dataset analysed and predicted with LogisticRegression function, Gridsearch tuned and with RFE

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    # Loads the data, cleans it, and perform one-hot encoding.
    #    Args:    filepath: The path to the CSV file.
    #    Returns: A preprocessed Pandas DataFrame.

    df = pd.read_csv(filepath)

    # Convert age ranges to midpoints
    age_mapping = {
        '0_10': 5, '10_20': 15, '20_30': 25, '30_40': 35,
        '40_50': 45, '50_60': 55, '60_70': 65, '70_80': 75,
        '80_90': 85, '90_100': 95, '100_110': 105
    }
    df['age'] = df['age'].replace(age_mapping).astype(np.int64)

    # Drop 'income' as it's not a meaningful predictor in this context
    df.drop(['income'], axis=1, inplace=True)

    # Perform one-hot encoding on categorical variables
    return pd.get_dummies(df)

def main():
    # Main function to run the COVID-19 predictive model analysis.
    
    # Data Loading and Preprocessing
    file_path = '../Data/Nexoid_COVID-19_data_subset.csv'
    df = load_and_preprocess_data(file_path)

    # Target/Input Split
    y = df['covid19_positive']
    X = df.drop(['covid19_positive'], axis=1)

    # Setting a random state for reproducibility
    random_state = 10
  
    X_mat = X.to_numpy()

    # Set as test siza 30%
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # initialise a standard scaler object
    scaler = StandardScaler()
        
    # visualise min, max, mean and standard dev of data before scaling
    print("Before scaling\n-------------")
    for i in range(5):
        col = X_train[:,i]
        print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
            format(i, min(col), max(col), np.mean(col), np.std(col)))

    # learn the mean and std.dev of variables from training data
    # then use the learned values to transform training data
    X_train_scaled = scaler.fit_transform(X_train, y_train)
   
    print("After scaling\n-------------")
    for i in range(5):
        col = X_train_scaled[:,i]
        print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
            format(i, min(col), max(col), np.mean(col), np.std(col)))

    # use the statistic from training to transform test data
    X_test_scaled = scaler.transform(X_test)

    
    # 01 - Base Model
    # Model Training and Evaluation
    print("\nBase Model\n----------")
    
    model = LogisticRegression(random_state=random_state)
    # fit model to training data
    model.fit(X_train_scaled, y_train)

    # training and test accuracy
    print("Train accuracy:", model.score(X_train_scaled, y_train))
    print("Test accuracy:", model.score(X_test_scaled, y_test))
    
    # classification report on test data
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # Feature Coefficients
    print("\nFeature Coefficient\n--------------------")
    coef = model.coef_[0]
    feature_names = X.columns
    sorted_features = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, importance in sorted_features:
        print(f"{name:<20}: {importance:.4f}")

    # Plot Coefficients
    coef = Series(model.coef_[0], index=X.columns).sort_values()
    coef.plot(kind='bar', title='Base Logistic Model Coefficients')
    plt.show()


    # 02 - Hyperparameter C Tuning with GridSearchCV
    print("\n\nGridSearchCV for Optimal Hyperparameters\n----------------------------------------")

    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=random_state),return_train_score=True, cv=10, n_jobs=-1)
    cv.fit(X_train_scaled, y_train)

   # Visualization of GridSearchCV Results
    result_set = cv.cv_results_
    print(result_set)
    
    train_result = result_set['mean_train_score']
    test_result = result_set['mean_test_score']
    print("Total number of models: ", len(test_result))
    # plot Hyperparameter C values vs training and test accuracy score
    plt.plot(range(0, len(train_result)), train_result, 'b', range(0,len(test_result)), test_result, 'r')
    plt.xlabel('Hyperparameter C\nBlue = training acc. Red = test acc.')
    plt.xticks(range(0, len(train_result)),[pow(10, x) for x in range(-6, 4)])
    plt.ylabel('score')
    plt.show()

    print("\nBest hyperparameters:\n---------------------\n")
    print(cv.best_params_)
    cv.fit(X_train_scaled, y_train)
    
    print("Train accuracy:", cv.score(X_train_scaled, y_train))
    print("Test accuracy:", cv.score(X_test_scaled, y_test))   
    
    # Gridsearch Plot
    # Get the best estimator from GridSearchCV run on the features
    selected_features = X.columns
    best_model = cv.best_estimator_
    # Create a Series with the coefficients and the feature names
    GridSearchCV_coef_series = Series(best_model.coef_[0], index=selected_features).sort_values()
    GridSearchCV_coef_series.plot(kind='bar', title='Model Coefficients for Gridsearch Features')
    plt.show() 

    
    # 03 - Recursive Feature Elimination with Cross Validation (RFECV)
    print("\nRecursive Feature Elimination with Cross-Validation\n-----------------------------\n")
    
    rfecv = RFECV(estimator = LogisticRegression(random_state=random_state), cv=10)
    rfecv.fit(X_train_scaled, y_train)

    # comparing how many variables before and after
    print("Original feature set", X_train_scaled.shape[1])
    print("Number of features after elimination", rfecv.n_features_)
    print("Selected features:\n------------------\n", X.columns[rfecv.support_])
 
    # Use selected features for Logistic Regression with tuned C
    X_train_rfe = rfecv.transform(X_train_scaled)
    X_test_rfe = rfecv.transform(X_test_scaled)

    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}

    # Re-run GridSearchCV with the reduced feature set
    print("\n GridSearchCV on RFE Selected")
    cv_rfe = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=random_state), cv=10, n_jobs=-1, return_train_score=True)
    cv_rfe.fit(X_train_rfe, y_train)

    # test the best model
    print("Train accuracy:", cv_rfe.score(X_train_rfe, y_train))
    print("Test accuracy:", cv_rfe.score(X_test_rfe, y_test))
    
    y_pred_rfe = cv_rfe.best_estimator_.predict(X_test_rfe)
    print("\nClassification Report (RFE):\n----------------------------\n", classification_report(y_test, y_pred_rfe))

    # print parameters of the best model
    print(cv_rfe.best_params_)  

    # Visualization of GridSearchCV Results (RFE)
    results_rfe = cv_rfe.cv_results_
    train_scores_rfe = results_rfe['mean_train_score']
    test_scores_rfe = results_rfe['mean_test_score']

    print("Total number of models: ", len(test_scores_rfe))
    # plot RFE model vs training and test accuracy score
    plt.plot(range(0, len(train_scores_rfe)), train_scores_rfe, 'b', range(0,len(test_scores_rfe)), test_scores_rfe, 'r')
    plt.xlabel('Hyperparameter C\nBlue = training acc. Red = test acc.')
    plt.xticks(range(0, len(train_scores_rfe)),[pow(10, x) for x in range(-6, 4)])
    plt.ylabel('score')
    plt.show()
 
    # Gridsearch RFE Plot
    # Get the best estimator from GridSearchCV run on the features
    selected_features = X.columns[rfecv.support_]
    best_model_rfe = cv_rfe.best_estimator_
    # Create a Series with the coefficients and the feature names
    GridSearchCV_rfe_coef_series = Series(best_model_rfe.coef_[0], index=selected_features).sort_values()
    GridSearchCV_rfe_coef_series.plot(kind='bar', title='Model Coefficients for Gridsearch RFE Features')
    plt.show()


    # Comparison of Models with ROC Curve
    print("\nComparing All Models with ROC Curve\n---------------------------------")
    y_pred_proba_base = model.predict_proba(X_test_scaled)
    y_pred_proba_tuned = best_model.predict_proba(X_test_scaled)
    y_pred_proba_rfe_tuned = best_model_rfe.predict_proba(X_test_rfe)

    roc_index_base = roc_auc_score(y_test, y_pred_proba_base[:, 1])
    roc_index_tuned = roc_auc_score(y_test, y_pred_proba_tuned[:, 1])
    roc_index_rfe_tuned = roc_auc_score(y_test, y_pred_proba_rfe_tuned[:, 1])

    print(f"ROC AUC for Base Model: {roc_index_base:.4f}")
    print(f"ROC AUC for Tuned Model: {roc_index_tuned:.4f}")
    print(f"ROC AUC for RFE+Tuned Model: {roc_index_rfe_tuned:.4f}")

    # Plotting ROC curves
    fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_base[:, 1])
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_pred_proba_tuned[:, 1])
    fpr_rfe, tpr_rfe, _ = roc_curve(y_test, y_pred_proba_rfe_tuned[:, 1])

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_base, tpr_base, label=f'Base Model (AUC = {roc_index_base:.2f})')
    plt.plot(fpr_tuned, tpr_tuned, label=f'Tuned Model (AUC = {roc_index_tuned:.2f})')
    plt.plot(fpr_rfe, tpr_rfe, label=f'RFE+Tuned Model (AUC = {roc_index_rfe_tuned:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()