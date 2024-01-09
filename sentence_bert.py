import os
import random
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
from sentence_transformers import SentenceTransformer

sns.set_theme()

RANDOM_STATE = 42
NUM_EPOCHS_LIST = [
    1000,
    1000
]

BATCH_SIZES = [
    512,
    1024
]

EARLY_STOPPING_PATIENCE = 3
OUTPUT_FOLDER = './artifacts'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

random.seed(RANDOM_STATE)
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
    # 'text_stft_spectrogram_mfccs_pitches',
    # 'text_stft_spectrogram_mfccs_pitches_energy'
]

model_names_list = [
    'all-mpnet-base-v2',
    'all-MiniLM-L6-v2'
]

basic_input_sizes = [
    768,
    384
]

activation_functions = [
    nn.ReLU(),
    nn.GELU()
]


class CustomClassifier(nn.Module):
    def __init__(self, basic_input_size, num_audio_processing_features, num_classes, activation_function):
        super(CustomClassifier, self).__init__()
        self.basic_input_size = basic_input_size
        self.num_audio_processing_features = num_audio_processing_features
        self.num_classes = num_classes

        print(f'Custom classifier input size: {basic_input_size + num_audio_processing_features}')

        self.fc1 = nn.Linear(basic_input_size + num_audio_processing_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.fc4 = nn.Linear(256, num_classes, bias=True)
        self.activation_function = activation_function

        self.text_only_classifier = nn.Linear(basic_input_size, num_classes, bias=True)

    def forward(self, x):
        if self.num_audio_processing_features == 0:  # text only
            output = self.text_only_classifier(x)
        else:
            x = self.fc1(x)
            x = self.activation_function(x)
            x = self.fc2(x)
            x = self.activation_function(x)
            x = self.fc3(x)
            x = self.activation_function(x)
            output = self.fc4(x)

        return output


for model_name, basic_input_size, NUM_EPOCHS, BATCH_SIZE in zip(model_names_list, basic_input_sizes, NUM_EPOCHS_LIST, BATCH_SIZES):
    for current_activation_function in activation_functions:
        model = SentenceTransformer(model_name, device=device.type)

        for dataset_description in dataset_description_list:
            CURRENT_CLASSIFIER_CHECKPOINT_FILE = f'{OUTPUT_FOLDER}/sentence_embedding_custom_classifier_{current_activation_function}_{model_name}_{dataset_description}.pth'
            columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
            current_df = df[columns_to_use]

            X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2, random_state=RANDOM_STATE, stratify=df.label)

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
            y_val_encoded = pd.Series(label_encoder.transform(y_val), index=y_val.index)
            y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index)

            num_classes = len(df.label.unique().tolist())
            num_audio_processing_features = X_train_numerical_features.shape[1]

            classifier = CustomClassifier(basic_input_size, num_audio_processing_features, num_classes, current_activation_function)
            classifier = classifier.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)

            print(f'############################ {model_name} {current_activation_function} {dataset_description} results ##################################')
            early_stopping_counter = 0
            best_loss = float('inf')
            train_losses_list = []
            val_losses_list = []
            f1_val_scores_list = []
            auc_val_scores_list = []
            start_time = time.time()
            for epoch in range(NUM_EPOCHS):
                print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
                classifier.train()
                train_loss = 0.0

                np.random.seed(epoch)

                shuffled_index = np.random.permutation(X_train.index)
                X_train_shuffled = X_train.reindex(shuffled_index)
                if X_train_numerical_features.shape[0] != 0:
                    X_train_numerical_features_shuffled = X_train_numerical_features.reindex(shuffled_index)
                else:
                    X_train_numerical_features_shuffled = X_train_numerical_features
                y_train_shuffled = y_train_encoded.reindex(shuffled_index)

                for start_idx in range(0, len(X_train_shuffled), BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, len(X_train_shuffled))
                    print(f'Train Batch {start_idx/BATCH_SIZE + 1}/{len(list(range(0, len(X_train_shuffled), BATCH_SIZE)))}')

                    batch_X_train_text_df = X_train_shuffled.iloc[start_idx:end_idx]
                    batch_y_train = y_train_shuffled.iloc[start_idx:end_idx]
                    batch_X_train_numerical_features_df = X_train_numerical_features_shuffled.iloc[start_idx:end_idx]

                    optimizer.zero_grad()

                    texts = batch_X_train_text_df['text'].tolist()

                    sentence_embeddings = model.encode(texts, convert_to_tensor=True if device.type == 'cuda' else False, batch_size=BATCH_SIZE)

                    if len(columns_to_use) > 1:
                        combined_features = torch.cat([sentence_embeddings, torch.tensor(batch_X_train_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                    else:
                        combined_features = torch.cat([sentence_embeddings], dim=1)

                    own_model_outputs = classifier(combined_features)
                    loss = criterion(own_model_outputs, torch.tensor(batch_y_train.tolist()).to(device))

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    del sentence_embeddings
                    del combined_features
                    torch.cuda.empty_cache()

                average_train_loss = train_loss / X_train_shuffled.shape[0]
                train_losses_list.append(train_loss)

                classifier.eval()
                val_loss = 0.0
                y_val_pred_all = []
                val_probabilities_all = []

                with torch.no_grad():
                    for start_idx in range(0, len(X_val), BATCH_SIZE):
                        print(f'Val Batch {start_idx / BATCH_SIZE + 1}/{len(list(range(0, len(X_val), BATCH_SIZE)))}')
                        end_idx = min(start_idx + BATCH_SIZE, len(X_val))

                        batch_X_val_df = X_val.iloc[start_idx:end_idx]
                        batch_y_val = y_val_encoded.iloc[start_idx:end_idx]
                        batch_X_val_numerical_features_df = X_val_numerical_features.iloc[start_idx:end_idx]

                        texts = batch_X_val_df['text'].tolist()
                        sentence_embeddings = model.encode(texts, convert_to_tensor=True if device.type == 'cuda' else False, batch_size=BATCH_SIZE)

                        if len(columns_to_use) > 1:
                            combined_features = torch.cat([sentence_embeddings, torch.tensor(batch_X_val_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                        else:
                            combined_features = torch.cat([sentence_embeddings], dim=1)

                        outputs = classifier(combined_features)
                        loss = criterion(outputs, torch.tensor(batch_y_val.tolist()).to(device))

                        val_loss += loss.item()

                        probabilities = torch.softmax(outputs, dim=1)

                        _, y_val_pred_encoded = torch.max(probabilities, 1)

                        if device.type == 'cuda':
                            y_val_pred_encoded = y_val_pred_encoded.cpu().detach()
                            val_probabilities_all.extend(probabilities.cpu().detach().numpy())
                        else:
                            y_val_pred_encoded = y_val_pred_encoded.detach()
                            val_probabilities_all.extend(probabilities.detach().numpy())

                        y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded.numpy())
                        y_val_pred_all.extend(y_val_pred)

                average_val_loss = val_loss / X_val.shape[0]
                val_losses_list.append(val_loss)
                f1 = f1_score(y_val, y_val_pred_all, average='macro')
                auc = roc_auc_score(y_val_encoded, val_probabilities_all, multi_class='ovr', average='macro')
                f1_val_scores_list.append(f1*100)
                auc_val_scores_list.append(auc*100)

                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.6f} - Average Training Loss: {average_train_loss:.6f} - Validation Loss: {val_loss:.6f} - Average Validation Loss: {average_val_loss:.6f} - Macro F1-Score: {f1*100:.6f} - AUC-ROC: {auc*100:.6f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0
                    torch.save({'epoch': epoch, 'classifier_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}, CURRENT_CLASSIFIER_CHECKPOINT_FILE)
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f'Early stopping triggered')
                    break

            end_time = time.time()

            # Load the best model again
            checkpoint = torch.load(CURRENT_CLASSIFIER_CHECKPOINT_FILE)
            epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
            print(f'Loaded best model from epoch: {epoch+1} and with loss: {best_loss}')

            # Save learning curve
            epochs = range(1, len(train_losses_list) + 1)
            plt.figure(figsize=(15, 5))
            plt.plot(epochs, train_losses_list, label='Training Loss', marker='.')
            plt.plot(epochs, val_losses_list, label='Validation Loss', marker='o')
            plt.plot(epochs, f1_val_scores_list, label='Macro F1-score', marker='x')
            plt.plot(epochs, auc_val_scores_list, label='AUC-ROC', marker='d')
            plt.title(f'{CURRENT_CLASSIFIER_CHECKPOINT_FILE[len(OUTPUT_FOLDER)+1:-4]} learning curve')
            plt.xlabel('Epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(f'{CURRENT_CLASSIFIER_CHECKPOINT_FILE[:-4]}_learning_curve.png')

            plt.show()

            # Save metrics on test set
            y_pred_all = []
            probabilities_all = []
            with torch.no_grad():
                for start_idx in range(0, len(X_test), BATCH_SIZE):
                    print(f'Test Batch {start_idx / BATCH_SIZE + 1}/{len(list(range(0, len(X_test), BATCH_SIZE)))}')
                    end_idx = min(start_idx + BATCH_SIZE, len(X_test))

                    batch_X_test_df = X_test.iloc[start_idx:end_idx]
                    batch_X_test_numerical_features_df = X_test_numerical_features.iloc[start_idx:end_idx]

                    texts = batch_X_test_df['text'].tolist()

                    sentence_embeddings = model.encode(texts, convert_to_tensor=True if device.type == 'cuda' else False, batch_size=BATCH_SIZE)

                    if len(columns_to_use) > 1:
                        combined_features = torch.cat([sentence_embeddings, torch.tensor(batch_X_test_numerical_features_df.values, dtype=torch.float32).to(device)], dim=1)
                    else:
                        combined_features = torch.cat([sentence_embeddings], dim=1)

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
            auc = roc_auc_score(y_test_encoded, probabilities_all, multi_class='ovr', average='macro')
            print(f"Total time for fitting: {end_time - start_time:.4f} seconds")
            print(f"Macro F1 Score dense layer: {f1*100:.6f}")
            print(classification_report(y_test, y_pred_all))
            print(f'Dense layer AUC-ROC: {auc*100:.6f}')

        del model
        torch.cuda.empty_cache()
