import time
import random

import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

df = pd.read_csv('data_processed.csv')
dataset_description_list = [
    #'text',
    #'text_stft',
    #'text_stft_spectrogram',
    'text_stft_spectrogram_mfccs',
    # 'text_stft_spectrogram_mfccs_pitches',
    # 'text_stft_spectrogram_mfccs_pitches_energy'
]

for dataset_description in dataset_description_list:
    columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
    current_df = df[columns_to_use]

    X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2,
                                                        random_state=RANDOM_STATE, stratify=df.label)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE,
                                                      stratify=y_train)

    # scaler = StandardScaler() # required for fast convergence of saga solver but TfidfVectorizer has norm='l2' by default so all weights per row are less than one
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train_features = tfidf_vectorizer.fit_transform(X_train.text)
    tfidf_train_df = pd.DataFrame(tfidf_train_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_test_features = tfidf_vectorizer.transform(X_test.text)
    tfidf_test_df = pd.DataFrame(tfidf_test_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    tfidf_train_df.reset_index(drop=True, inplace=True)
    tfidf_test_df.reset_index(drop=True, inplace=True)

    if len(columns_to_use) > 1:
        X_train = pd.concat([X_train.drop(columns=['text']), tfidf_train_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=['text']), tfidf_test_df], axis=1)
    else:
        X_train = tfidf_train_df
        X_test = tfidf_test_df

    def plot_histogram(coefficients, group_name):
        plt.hist(coefficients, bins=20, edgecolor='black')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Coefficients for {group_name}')
        plt.show()

    print(f'############################ {dataset_description} results ##################################')

    # Logistic regression
    model_filename = "logistic_regression_" + dataset_description +".joblib"
    logistic_regression = joblib.load(model_filename)
    y_pred = logistic_regression.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score logistic regression: {f1}")
    print(classification_report(y_test, y_pred))
    print('Logistic regression AUC-ROC: ' + str(roc_auc_score(y_test, logistic_regression.predict_proba(X_test), multi_class='ovr', average='macro')))

    # Create a dictionary to store feature importance for each group
    feature_importance_by_group = {}
    coefficients_by_group = {}
    feature_categories = [category for category in dataset_description.split('_')]

    print(f"\nFeature categories")
    print(feature_categories)
    # Iterate through feature categories/groups
    for feature_category in feature_categories:
        # Filter features belonging to the current group

        if feature_category == 'text':
            group_features = [feature for feature in X_train.columns if not any(feature.startswith(other_category + "_")
                                                                                for other_category in feature_categories
                                                                                if other_category != 'text')]
        else:
            group_features = [feature for feature in X_train.columns if feature.startswith(f"{feature_category}_")]

        # Extract coefficients of features in the current group
        group_coefficients = logistic_regression.coef_[:,
                             [X_train.columns.get_loc(feature) for feature in group_features]]

        # Calculate feature importance for the current group
        feature_weight = len(group_features)/(X_train.shape[1])
        group_importance = feature_weight * np.abs(group_coefficients).mean()

        # group_importance = np.abs(group_coefficients).sum()
        # group_importance = np.abs(group_coefficients).mean()

        coefficients_by_group[feature_category] = group_coefficients
        # Store the importance value in the dictionary
        feature_importance_by_group[feature_category] = group_importance

    # Print or analyze feature importance by group
    for feature_category, importance in feature_importance_by_group.items():
        print(f"Feature Category: {feature_category}, Importance: {importance}")

    # for feature_category in coefficients_by_group:
        # Create and plot the histogram for the group
        #plot_histogram(coefficients_by_group[feature_category], feature_category)

    # Extract the feature category names and corresponding importance values
    category_names = list(feature_importance_by_group.keys())
    importance_values = list(feature_importance_by_group.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.barh(category_names, importance_values, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature Categories')
    plt.title('Feature Importance by Group')

    # Show the plot
    plt.show()


