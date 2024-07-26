import sys
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import ast
from sklearn.preprocessing import LabelEncoder


# Dataset selection
dataset_selection = sys.argv[1]

df1=pd.read_excel(r"datasets\multi label dataset of small talk in smart home (2).xlsx")
df1=df1[['Voice', 'Context']]

df2=pd.read_excel(r"datasets\dataset_refined_context.xlsx")
df2=df2[['Voice', 'Context']]

if dataset_selection == "1":
    df=df1

elif dataset_selection == "2":
    df=df2

elif dataset_selection == "3" or dataset_selection == "4":
    frames = [df1, df2]  
    df = pd.concat(frames)

print("Dataset:")
print(df.head())

# Separate features (text) and labels 
X = df['Voice']
y = df['Context']

#Literal eval
y=y.apply(lambda x: ast.literal_eval(x))

# encode string labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str))  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Context'].unique()))

# Tokenize and convert text data to input tensors
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')

# Convert labels to tensors with the correct data type
train_labels = torch.tensor(y_train, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop (fine-tuning BERT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Calculate class frequencies and class weights
class_frequencies = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_frequencies) * class_frequencies.astype(float))

class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}


epochs = 2
best_weighted_f1 = 0.0
patience = 1
num_epochs_without_improvement = 0

if dataset_selection == "4":
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Apply class weights to the loss function
            loss_weights = torch.tensor([class_weights_dict[i.item()] for i in labels], dtype=torch.float).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = (loss_weights * outputs.loss).mean()
            loss.backward()
            optimizer.step()

        # Evaluation on the training set after each epoch
        model.eval()
        train_predictions = []
        train_true_labels = []

        with torch.no_grad():
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)

                train_predictions.extend(predicted_labels.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())

        train_weighted_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        print(f"Epoch {epoch+1} - Training Weighted F1 Score: {train_weighted_f1}")

        # Evaluation on the test set after each epoch
        model.eval()
        test_predictions = []
        test_true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)

                test_predictions.extend(predicted_labels.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

        test_weighted_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
        print(f"Epoch {epoch+1} - Testing Weighted F1 Score: {test_weighted_f1}")

        # Check for early stopping
        if test_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = test_weighted_f1
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

else:
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Evaluation on the training set after each epoch
        model.eval()
        train_predictions = []
        train_true_labels = []

        with torch.no_grad():
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)

                train_predictions.extend(predicted_labels.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())

        train_weighted_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        print(f"Epoch {epoch+1} - Training Weighted F1 Score: {train_weighted_f1}")

        # Evaluation on the test set after each epoch
        model.eval()
        test_predictions = []
        test_true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)

                test_predictions.extend(predicted_labels.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

        test_weighted_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
        print(f"Epoch {epoch+1} - Testing Weighted F1 Score: {test_weighted_f1}")

        # Check for early stopping
        if test_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = test_weighted_f1
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break


# Evaluation on the test set
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# # Calculate accuracy and print the classification report
# accuracy = accuracy_score(true_labels, predictions)
# print("Accuracy:", accuracy)

# report = classification_report(true_labels, predictions)
# print("Classification Report:\n", report)

weighted_f1 = f1_score(true_labels, predictions, average='weighted')
print("Weighted F1 Score:", weighted_f1)