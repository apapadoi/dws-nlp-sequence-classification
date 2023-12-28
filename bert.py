import os
import time
import random

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
from transformers import AutoTokenizer, BertForSequenceClassification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

sns.set_theme()

RANDOM_STATE = 42
NUM_EPOCHS = [3, 2] # TODO state in paper for how many epochs we fine tune
BATCH_SIZES = [
    32,
    10
]
BATCH_SIZE_MULTIPLIERS = [
    10,
    16
]

EARLY_STOPPING_PATIENCE = 3
DESIRED_SAMPLES_PER_CLASS = 100
OUTPUT_FOLDER = './artifacts'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU available: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

df = pd.read_csv('data_processed.csv')
NUM_LABELS=len(df.label.unique().tolist())

dataset_description_list = [
    'text',
]

model_names_list = [
    'bert-base-uncased',
    'bert-large-uncased'
]

# TODO try sentence BERT with logistic regression or linear layer after pooling step in order to use also the other features as in baselines_and_logistic_regression.py \
#  also see bertforsequenceclassification configuration and mimic this and try the best model of sentence-transformers and a smaller model as well to see if we can achieve \
#  same performance with less parameters
for model_name, BATCH_SIZE, BATCH_SIZE_MULTIPLIER, NUM_EPOCHS in zip(model_names_list, BATCH_SIZES, BATCH_SIZE_MULTIPLIERS, NUM_EPOCHS):
    for dataset_description in dataset_description_list:
        columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
        current_df = df[columns_to_use]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2,
                                                            random_state=RANDOM_STATE, stratify=df.label)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)

        # rus = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy={key: DESIRED_SAMPLES_PER_CLASS if y_train.value_counts().loc[key] > DESIRED_SAMPLES_PER_CLASS else y_train.value_counts().loc[key] for key in y_train.unique().tolist()})
        # X_train, y_train = rus.fit_resample(X_train, y_train)
        # ros = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy={key: DESIRED_SAMPLES_PER_CLASS if y_train.value_counts().loc[key] < DESIRED_SAMPLES_PER_CLASS else y_train.value_counts().loc[key] for key in y_train.unique().tolist()})
        # X_train, y_train = ros.fit_resample(X_train, y_train)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_val_encoded = pd.Series(label_encoder.fit_transform(y_val), index=y_val.index)
        y_test_encoded = pd.Series(label_encoder.fit_transform(y_test), index=y_test.index)

        max_len = 512

        def tokenize_text(text):
            return tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )


        class CustomDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                tokenized_text = tokenize_text(text)
                return {
                    'input_ids': tokenized_text['input_ids'].squeeze(0),
                    'attention_mask': tokenized_text['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                }


        train_dataset = CustomDataset(X_train.text, y_train_encoded)
        val_dataset = CustomDataset(X_val.text, y_val_encoded)
        test_dataset = CustomDataset(X_test.text, y_test_encoded)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_MULTIPLIER*BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_MULTIPLIER*BATCH_SIZE, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs")
            model = torch.nn.parallel.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        early_stopping_counter = 0
        best_loss = float('inf')
        early_stopping_epoch = -1
        train_losses_list = []
        val_losses_list = []
        start_time = time.time()

        print(f'############################ {model_name} {dataset_description} results ##################################')

        for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
            model.train()
            train_loss = 0
            for batch_index, batch in enumerate(train_loader):
                print(f'Train Batch {batch_index+1}/{len(train_loader)}')
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if torch.cuda.device_count() > 1:
                    train_loss += loss.sum()
                    loss.sum().backward()
                else:
                    train_loss += loss
                    loss.backward()

                optimizer.step()

            if device.type == 'cuda':
                train_losses_list.append(train_loss.cpu().detach().numpy())
            else:
                train_losses_list.append(train_loss.detach().numpy())

            del outputs
            torch.cuda.empty_cache()

            # Evaluation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    print(f'Validation Batch {batch_index + 1}/{len(val_loader)}')
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if torch.cuda.device_count() > 1:
                        val_loss += loss.sum()
                    else:
                        val_loss += loss

            del outputs
            torch.cuda.empty_cache()

            if device.type == 'cuda':
                val_losses_list.append(val_loss.cpu().detach().numpy())
            else:
                val_losses_list.append(val_loss.detach().numpy())

            print(
                f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.5f} - Average Training Loss: {train_loss/X_train.shape[0]:.5f} - Validation Loss: {val_loss:.5f} - Average Validation Loss: {val_loss/X_val.shape[0]}')

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
        torch.save(model.state_dict(), f'{OUTPUT_FOLDER}/{model_name}_{dataset_description}_epoch_{early_stopping_epoch+1 if early_stopping_epoch != -1 else NUM_EPOCHS}_out_of_{NUM_EPOCHS}.pth')
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
        model.eval()
        y_pred_all = []
        probabilities_all = []
        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                print(f'Test Batch {batch_index + 1}/{len(test_loader)}')
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                probabilities = torch.softmax(outputs.logits, dim=1)

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

        del outputs
        del model
        del tokenizer
        torch.cuda.empty_cache()
