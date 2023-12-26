import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from torch.optim import lr_scheduler

sns.set_theme()

RANDOM_STATE = 42
NUM_EPOCHS = 100 # TODO state in paper that the max allowed epochs are 100
BATCH_SIZE = 1024
EARLY_STOPPING_PATIENCE = 10
DESIRED_SAMPLES_PER_CLASS = 500
OUTPUT_FOLDER = './artifacts'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

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
    # 'text_stft',
    # 'text_stft_spectrogram',
    # 'text_stft_spectrogram_mfccs',
    # 'text_stft_spectrogram_mfccs_pitches',
    # 'text_stft_spectrogram_mfccs_pitches_energy'
]

model_names_list = [
    'bert-base-uncased',
    'bert-large-uncased'
]

basic_input_sizes = [
    768,
    1024
]

# TODO try logistic regression after pooling step
# TODO try BertForSequenceClassification and might reject idea about audio processing features for BERT
# TODO try sentence BERT
class CustomClassifier(nn.Module):
    def __init__(self, basic_input_size, num_audio_processing_features, num_classes):
        super(CustomClassifier, self).__init__()
        print(f'Custom classifier input size: {basic_input_size + num_audio_processing_features}')
        self.fc1 = nn.Linear(basic_input_size + num_audio_processing_features, num_classes)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(basic_input_size + num_audio_processing_features, num_classes)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        output = self.fc1(x)
        output = self.softmax(output)
        return output


for model_name, basic_input_size in zip(model_names_list, basic_input_sizes):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    for dataset_description in dataset_description_list:
        columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
        current_df = df[columns_to_use]

        # rus = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy={key: DESIRED_SAMPLES_PER_CLASS if df.label.value_counts().loc[key] > DESIRED_SAMPLES_PER_CLASS else df.label.value_counts().loc[key] for key in df.label.unique().tolist()})
        # current_df, y = rus.fit_resample(current_df, df.label)
        # ros = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy={key: DESIRED_SAMPLES_PER_CLASS if y.value_counts().loc[key] < DESIRED_SAMPLES_PER_CLASS else y.value_counts().loc[key] for key in y.unique().tolist()})
        # current_df, y = ros.fit_resample(current_df, y)

        X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2,
                                                            random_state=RANDOM_STATE, stratify=df.label)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)

        if len(columns_to_use) > 1:
            X_train_numerical_features = X_train.drop(columns='text')
            X_val_numerical_features = X_val.drop(columns='text')
            X_test_numerical_features = X_test.drop(columns='text')
        else:
            X_train_numerical_features = pd.DataFrame([])
            X_val_numerical_features = pd.DataFrame([])
            X_test_numerical_features = pd.DataFrame([])

        label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_val_encoded = pd.Series(label_encoder.fit_transform(y_val), index=y_val.index)
        y_test_encoded = pd.Series(label_encoder.fit_transform(y_test), index=y_test.index)

        num_classes = len(df.label.unique().tolist())
        num_audio_processing_features = X_train_numerical_features.shape[1]

        classifier = CustomClassifier(basic_input_size, num_audio_processing_features, num_classes)
        classifier = classifier.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=33)

        print(f'############################ {model_name} {dataset_description} results ##################################')
        early_stopping_counter = 0
        best_loss = float('inf')
        early_stopping_epoch = -1
        train_losses_list = []
        val_losses_list = []
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            classifier.train()
            train_loss = 0.0

            np.random.seed(epoch)

            shuffled_index = np.random.permutation(X_train.index)
            X_train_shuffled = X_train.reindex(shuffled_index)
            X_train_numerical_features_shuffled = X_train_numerical_features.reindex(shuffled_index)
            y_train_shuffled = y_train_encoded.reindex(shuffled_index)

            for start_idx in range(0, len(X_train_shuffled), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_train_shuffled))

                batch_X_train_text_df = X_train_shuffled.iloc[start_idx:end_idx]
                batch_y_train = y_train_shuffled.iloc[start_idx:end_idx]
                batch_X_train_numerical_features_df = X_train_numerical_features_shuffled.iloc[start_idx:end_idx]

                optimizer.zero_grad()

                texts = batch_X_train_text_df['text'].tolist()
                tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**tokenized_texts)
                    cls_embeddings = outputs[1]

                if len(columns_to_use) > 1:
                    combined_features = torch.cat([cls_embeddings, torch.tensor(batch_X_train_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                else:
                    combined_features = torch.cat([cls_embeddings], dim=1)


                own_model_outputs = classifier(combined_features)
                loss = criterion(own_model_outputs, torch.tensor(batch_y_train.tolist()).to(device))

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * combined_features.size(0)

                del cls_embeddings
                del combined_features

                torch.cuda.empty_cache()

            average_train_loss = train_loss / X_train_shuffled.shape[0]
            train_losses_list.append(train_loss)

            classifier.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for start_idx in range(0, len(X_val), BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, len(X_val))

                    batch_X_val_df = X_val.iloc[start_idx:end_idx]
                    batch_y_val = y_val_encoded.iloc[start_idx:end_idx]
                    batch_X_val_numerical_features_df = X_val_numerical_features.iloc[start_idx:end_idx]

                    texts = batch_X_val_df['text'].tolist()
                    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

                    outputs = model(**tokenized_texts)
                    cls_embeddings = outputs[1]

                    if len(columns_to_use) > 1:
                        combined_features = torch.cat([cls_embeddings, torch.tensor(batch_X_val_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                    else:
                        combined_features = torch.cat([cls_embeddings], dim=1)

                    outputs = classifier(combined_features)
                    loss = criterion(outputs, torch.tensor(batch_y_val.tolist()).to(device))

                    val_loss += loss.item() * combined_features.size(0)
                    total_samples += combined_features.size(0)

            average_val_loss = val_loss / total_samples
            val_losses_list.append(val_loss)
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.5f} - Average Training Loss: {average_train_loss:.5f} - Validation Loss: {val_loss:.5f} - Average Validation Loss: {average_val_loss:.5f}')

            scheduler.step()
            print(f'Updating learning rate to {optimizer.param_groups[0]["lr"]}')

            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f'Early stopping - Epoch: {epoch + 1}/{NUM_EPOCHS}, Best loss: {best_loss}')
                    early_stopping_epoch = epoch
                    break

        end_time = time.time()
        torch.save(classifier.state_dict(), f'{OUTPUT_FOLDER}/{model_name}_{dataset_description}_epoch_{early_stopping_epoch+1 if early_stopping_epoch != -1 else NUM_EPOCHS}_out_of_{NUM_EPOCHS}.pth')

        # Save learning curve
        epochs = range(1, len(train_losses_list) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses_list, label='Training Loss', marker='o')
        plt.plot(epochs, val_losses_list, label='Validation Loss', marker='o')
        plt.title(f'{model_name}_{dataset_description}_epoch_{early_stopping_epoch+1 if early_stopping_epoch != -1 else NUM_EPOCHS}/{NUM_EPOCHS} learning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f'{OUTPUT_FOLDER}/{model_name}_{dataset_description}_epoch_{early_stopping_epoch+1 if early_stopping_epoch != -1 else NUM_EPOCHS}_out_of_{NUM_EPOCHS}_learning_curve.png')

        plt.show()

        # Save metrics on test set
        classifier.eval()
        y_pred_all = []
        probabilities_all = []
        with torch.no_grad():
            for start_idx in range(0, len(X_test), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_test))

                batch_X_test_df = X_test.iloc[start_idx:end_idx]
                batch_X_test_numerical_features_df = X_test_numerical_features.iloc[start_idx:end_idx]

                texts = batch_X_test_df['text'].tolist()
                tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

                outputs = model(**tokenized_texts)
                cls_embeddings = outputs[1]

                if len(columns_to_use) > 1:
                    combined_features = torch.cat([cls_embeddings, torch.tensor(batch_X_test_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                else:
                    combined_features = torch.cat([cls_embeddings], dim=1)

                outputs = classifier(combined_features)

                probabilities = torch.softmax(outputs, dim=1)

                _, y_pred_encoded = torch.max(probabilities, 1)

                if device.type == 'cuda':
                    y_pred_encoded = y_pred_encoded.cpu().detach()
                    probabilities_all.extend(probabilities.cpu().detach().numpy())
                else:
                    y_pred_encoded = y_pred_encoded.detach()
                    probabilities_all.extend(probabilities.detach().numpy())

                y_pred = label_encoder.inverse_transform(y_pred_encoded.numpy())
                y_pred_all.extend(y_pred)


        f1 = f1_score(y_test, y_pred_all, average='macro')
        print(f"Total time for fitting: {end_time - start_time:.4f} seconds")
        print(f"Macro F1 Score dense layer: {f1}")
        print(classification_report(y_test, y_pred_all))
        print('Dense layer AUC-ROC: ' + str(roc_auc_score(y_test_encoded, probabilities_all, multi_class='ovr', average='macro')))
        # exit(0)
    del model
    del tokenizer
    torch.cuda.empty_cache()
