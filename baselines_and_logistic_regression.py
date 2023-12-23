import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

df = pd.read_csv('data_processed.csv')
dataset_description_list = [
    # 'text',
    # 'text_stft',
    # 'text_stft_spectrogram',
    # 'text_stft_spectrogram_mfccs',
    # 'text_stft_spectrogram_mfccs_pitches',
    'text_stft_spectrogram_mfccs_pitches_energy'
]

for dataset_description in dataset_description_list:
    columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
    current_df = df[columns_to_use]
    X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2, random_state=RANDOM_STATE, stratify=df.label)
    scaler = StandardScaler() # required for fast convergence of saga solver
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

    x_train_columns = X_train.columns
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=x_train_columns)

    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=x_train_columns)

    print(f'############################ {dataset_description} results ##################################')
    # Majority Baseline
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    y_pred = dummy_majority.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score majority baseline: {f1}")
    print(classification_report(y_test, y_pred))
    print('Majority baseline AUC-ROC: ' + str(roc_auc_score(y_test, dummy_majority.predict_proba(X_test), multi_class='ovr', average='macro')))

    # Stratified Baseline
    dummy_stratified = DummyClassifier(strategy='stratified')
    dummy_stratified.fit(X_train, y_train)
    y_pred = dummy_stratified.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score stratified baseline: {f1}")
    print(classification_report(y_test, y_pred))
    print('Stratified baseline AUC-ROC: ' + str(roc_auc_score(y_test, dummy_stratified.predict_proba(X_test), multi_class='ovr', average='macro')))

    # Random Baseline
    dummy_random = DummyClassifier(strategy='uniform')
    dummy_random.fit(X_train, y_train)
    y_pred = dummy_random.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score random baseline: {f1}")
    print(classification_report(y_test, y_pred))
    print('Random baseline AUC-ROC: ' + str(roc_auc_score(y_test, dummy_random.predict_proba(X_test), multi_class='ovr', average='macro')))

    # Logistic regression
    logistic_regression = LogisticRegression(n_jobs=14, random_state=RANDOM_STATE, max_iter=1000)
    start_time = time.time()
    logistic_regression.fit(X_train, y_train)
    end_time = time.time()
    y_pred = logistic_regression.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Total time for fitting: {end_time - start_time:.4f} seconds")
    print(f"Macro F1 Score logistic regression: {f1}")
    print(classification_report(y_test, y_pred))
    print('Logistic regression AUC-ROC: ' + str(roc_auc_score(y_test, logistic_regression.predict_proba(X_test), multi_class='ovr', average='macro')))
