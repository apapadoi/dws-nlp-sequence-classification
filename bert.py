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
MAX_LEN = 512
NUM_EPOCHS_LIST = [
    1000,
    1000
]

BATCH_SIZES = [
    32,
    10
]

BATCH_SIZE_MULTIPLIERS = [
    10,
    16
]

EARLY_STOPPING_PATIENCE = 3
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
NUM_LABELS = len(df.label.unique().tolist())

dataset_description_list = [
    'text',
]

model_names_list = [
    'bert-base-uncased',
    'bert-large-uncased'
]


for model_name, BATCH_SIZE, BATCH_SIZE_MULTIPLIER, NUM_EPOCHS in zip(model_names_list, BATCH_SIZES, BATCH_SIZE_MULTIPLIERS, NUM_EPOCHS_LIST):
    for dataset_description in dataset_description_list:
        CURRENT_CLASSIFIER_CHECKPOINT_FILE = f'{OUTPUT_FOLDER}/bert_for_seq_classification_{model_name}_{dataset_description}.pth'
        columns_to_use = [column for column in df.columns for feature_category in dataset_description.split('_') if feature_category in column]
        current_df = df[columns_to_use]

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        X_train, X_test, y_train, y_test = train_test_split(current_df, df.label, test_size=0.2, random_state=RANDOM_STATE, stratify=df.label)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_val_encoded = pd.Series(label_encoder.transform(y_val), index=y_val.index)
        y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index)

        def tokenize_text(text):
            return tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt' if device.type == 'cuda' else False,
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        print(f'############################ {model_name} {dataset_description} results ##################################')

        early_stopping_counter = 0
        best_loss = float('inf')
        train_losses_list = []
        val_losses_list = []
        f1_val_scores_list = []
        auc_val_scores_list = []
        start_time = time.time()

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

                train_loss += loss.item()
                loss.backward()

                optimizer.step()
                del outputs

            average_train_loss = train_loss / X_train.shape[0]
            train_losses_list.append(train_loss)

            torch.cuda.empty_cache()

            # Evaluation
            model.eval()
            val_loss = 0.0
            y_val_pred_all = []
            val_probabilities_all = []
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    print(f'Val Batch {batch_index + 1}/{len(val_loader)}')
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs.logits, dim=1)

                    _, y_val_pred_encoded = torch.max(probabilities, 1)

                    if device.type == 'cuda':
                        y_val_pred_encoded = y_val_pred_encoded.cpu().detach()
                        val_probabilities_all.extend(probabilities.cpu().detach().numpy())
                    else:
                        y_val_pred_encoded = y_val_pred_encoded.detach()
                        val_probabilities_all.extend(probabilities.detach().numpy())

                    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded.numpy())
                    y_val_pred_all.extend(y_val_pred)

            del outputs
            torch.cuda.empty_cache()

            average_val_loss = val_loss / X_val.shape[0]
            val_losses_list.append(val_loss)
            f1 = f1_score(y_val, y_val_pred_all, average='macro')
            auc = roc_auc_score(y_val_encoded, val_probabilities_all, multi_class='ovr', average='macro')
            f1_val_scores_list.append(f1 * 100)
            auc_val_scores_list.append(auc * 100)


            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.6f} - Average Training Loss: {average_train_loss:.6f} - Validation Loss: {val_loss:.6f} - Average Validation Loss: {average_val_loss:.6f} - Macro F1-Score: {f1*100:.6f} - AUC-ROC: {auc*100:.6f}')

            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
                torch.save({'epoch': epoch, 'classifier_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}, CURRENT_CLASSIFIER_CHECKPOINT_FILE)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered')
                break

        end_time = time.time()

        # Load the best model again
        checkpoint = torch.load(CURRENT_CLASSIFIER_CHECKPOINT_FILE)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        print(f'Loaded best model from epoch: {epoch + 1} and with loss: {best_loss}')

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
        auc = roc_auc_score(y_test_encoded, probabilities_all, multi_class='ovr', average='macro')
        print(f"Total time for fitting: {end_time - start_time:.4f} seconds")
        print(f"Macro F1 Score dense layer: {f1*100:.6f}")
        print(classification_report(y_test, y_pred_all))
        print(f'Dense layer AUC-ROC: {auc*100:.6f}')

        del outputs
        del model
        del tokenizer
        torch.cuda.empty_cache()
