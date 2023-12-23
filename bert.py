import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
NUM_EPOCHS = 100
BATCH_SIZE = 1024
BASIC_INPUT_SIZE = 768

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU available: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

df = pd.read_csv('data_processed.csv')
dataset_description_list = [
    'text',
    'text_stft',
    'text_stft_spectrogram',
    'text_stft_spectrogram_mfccs',
    'text_stft_spectrogram_mfccs_pitches',
    'text_stft_spectrogram_mfccs_pitches_energy'
]

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model = model.to(device)

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)


class CustomClassifier(nn.Module):
    def __init__(self, num_audio_processing_features, num_classes):
        super(CustomClassifier, self).__init__()
        print(f'Custom classifier input size: {BASIC_INPUT_SIZE + num_audio_processing_features}')
        self.fc1 = nn.Linear(BASIC_INPUT_SIZE + num_audio_processing_features, num_classes)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.relu = nn.ReLU()
        # self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        output = self.fc1(x)
        output = self.softmax(output)  # Applying softmax for obtaining class probabilities
        return output


for dataset_description in dataset_description_list:
    columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
    current_df = df[columns_to_use]

    X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2,
                                                        random_state=RANDOM_STATE, stratify=df.label)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)

    if len(columns_to_use) > 1:
        X_train_numerical_features = X_train.drop(columns='text')
        X_val_numerical_features = X_val.drop(columns='text')
    else:
        X_train_numerical_features = pd.DataFrame([])
        X_val_numerical_features = pd.DataFrame([])

    label_encoder = LabelEncoder()
    y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
    y_val_encoded = pd.Series(label_encoder.fit_transform(y_val), index=y_val.index)

    num_classes = len(df.label.unique().tolist())
    num_audio_processing_features = X_train_numerical_features.shape[1]

    classifier = CustomClassifier(num_audio_processing_features, num_classes)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    print(f'############################ {dataset_description} results ##################################')
    # Bert
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        train_loss = 0.0

        np.random.seed(epoch)

        X_train_shuffled = X_train.sample(frac=1, random_state=RANDOM_STATE)
        X_train_numerical_features_shuffled = X_train_numerical_features.sample(frac=1, random_state=RANDOM_STATE)
        y_train_shuffled = y_train_encoded.sample(frac=1, random_state=RANDOM_STATE)

        for start_idx in range(0, len(X_train_shuffled), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(X_train_shuffled))

            batch_X_train_text_df = X_train_shuffled.iloc[start_idx:end_idx]
            batch_y_train = y_train_shuffled.iloc[start_idx:end_idx]
            batch_X_train_numerical_features_df = X_train_numerical_features_shuffled.iloc[start_idx:end_idx]

            texts = batch_X_train_text_df['text'].tolist()
            tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**tokenized_texts)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            if len(columns_to_use) > 1:
                combined_features = torch.cat([cls_embeddings, torch.tensor(batch_X_train_numerical_features_df.values).to(device)], dim=1)
            else:
                combined_features = torch.cat([cls_embeddings], dim=1)

            optimizer.zero_grad()
            own_model_outputs = classifier(combined_features)
            loss = criterion(own_model_outputs, torch.tensor(batch_y_train.tolist()).to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * combined_features.size(0)

            del cls_embeddings
            del combined_features

            torch.cuda.empty_cache()

        average_train_loss = train_loss / X_train_shuffled.shape[0]

        classifier.eval()
        val_loss = 0.0
        total_samples = 0
        # print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {average_train_loss:.4f}')
        with torch.no_grad():
            for start_idx in range(0, len(X_val), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_val))

                batch_X_val_df = X_val.iloc[start_idx:end_idx]
                batch_y_val = y_val_encoded.iloc[start_idx:end_idx]
                batch_X_val_numerical_features_df = X_val_numerical_features.iloc[start_idx:end_idx]

                texts = batch_X_val_df['text'].tolist()
                tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**tokenized_texts)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]

                if len(columns_to_use) > 1:
                    combined_features = torch.cat([cls_embeddings, torch.tensor(batch_X_val_numerical_features_df.values).to(device)], dim=1)
                else:
                    combined_features = torch.cat([cls_embeddings], dim=1)

                outputs = classifier(combined_features)
                loss = criterion(outputs, torch.tensor(batch_y_val.tolist()).to(device))

                val_loss += loss.item() * combined_features.size(0)
                total_samples += combined_features.size(0)
        #
        average_val_loss = val_loss / total_samples
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}')
    # classifier.eval()
    # TODO add metrics on test set - save model and parameters - save learning curves - add early stopping with patience 10 and state in paper that the max allowed epochs are 100 - add experiment with bert-large
    # with torch.no_grad():
        # predictions = classifier(torch.tensor(X_test))

    # Get predicted class labels by taking the argmax of the probabilities
    # predicted_labels = torch.argmax(predictions, dim=1).tolist()
    # print("Predicted labels:", predicted_labels)
    # logreg_model = LogisticRegression(n_jobs=14, random_state=RANDOM_STATE, solver='saga')
    # start_time = time.time()
    # logreg_model.fit(X_train, y_train)
    # end_time = time.time()
    # y_pred = logreg_model.predict(X_test)
    #
    # f1 = f1_score(y_test, y_pred, average='macro')
    # print(f"Total time for fitting: {end_time - start_time:.4f} seconds")
    # print(f"F1 Score logistic regression: {f1}")
    # print(classification_report(y_test, y_pred))
