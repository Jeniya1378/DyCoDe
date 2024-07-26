import sys
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Concatenate, Embedding, Attention
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1_val = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1_val

# Dataset selection
# Check if a value is passed as a command-line argument
if len(sys.argv) > 1:
    dataset_selection = sys.argv[1]
else:
    # Set a default value here if no argument is passed
    dataset_selection = "4"

df1 = pd.read_excel(r"datasets\multi label dataset of small talk in smart home (2).xlsx")
df1 = df1[['Voice', 'Context']]

df2 = pd.read_excel(r"datasets\dataset_refined_context.xlsx")
df2 = df2[['Voice', 'Context']]

if dataset_selection == "1":
    df = df1
elif dataset_selection == "2":
    df = df2
elif dataset_selection == "3" or dataset_selection == "4":
    frames = [df1, df2]
    df = pd.concat(frames)

print("Dataset:")
print(df.head())

# Converting to lowercase
df['Voice'] = df['Voice'].str.lower()

# Filtering the rare terms.
df = df.groupby("Context", sort=False).filter(lambda x: len(x) > 1)
df.shape

# Literal evaluate
df['Context'] = df['Context'].apply(lambda x: ast.literal_eval(x))

# Label Binarization
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Context'])
numOfClasses = len(multilabel.classes_)

y = pd.DataFrame(y, columns=multilabel.classes_)
Y = y[[x for x in multilabel.classes_]]
np.asarray(Y).astype('float32').reshape((-1, 1))

# Encoding voice sample to feed in the network
arr = df['Voice'].to_numpy()
modelSBERT = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = modelSBERT.encode(arr)

# Train test split
train1_x, test_x, train1_y, test_y = train_test_split(sentence_embeddings,
                                                      Y,
                                                      train_size=0.7,
                                                      test_size=0.3,
                                                      random_state=42)

# Attention Layer
query_input = tf.keras.Input(shape=(384,), dtype='float32')
value_input = tf.keras.Input(shape=(384,), dtype='float32')
token_embedding = tf.keras.layers.Embedding(input_dim=43920, output_dim=384)
query_embeddings = token_embedding(query_input)
value_embeddings = token_embedding(value_input)

cnn_layer = tf.keras.layers.Conv1D(
    filters=192,
    kernel_size=4,
    padding='same')
query_seq_encoding = cnn_layer(query_embeddings)
value_seq_encoding = cnn_layer(value_embeddings)
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)
input_layer1 = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

class_frequencies = np.sum(train1_y, axis=0)
total_samples = len(train1_y)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)

class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}

input_layer = Input(shape=(384,), tensor=input_layer1)
Layer_1 = Dense(512, activation="relu")(input_layer)
Layer_2 = Dense(256, activation="relu")(Layer_1)
output_layer = Dense(numOfClasses, activation="sigmoid")(Layer_2)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy", "binary_accuracy", f1_score_metric])

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(test_x) > 0.5).astype("int32")
        f1 = f1_score(test_y, y_pred, average='weighted')
        print(f"Weighted F1 Score for epoch {epoch + 1}: {f1}")

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
f1_callback = F1ScoreCallback()

if dataset_selection == "4":
    trained_model = model.fit(train1_x, train1_y, class_weight=class_weights_dict, epochs=100, batch_size=128, validation_data=(test_x, test_y), callbacks=[callback])
else:
    trained_model = model.fit(train1_x, train1_y, epochs=100, batch_size=128, validation_data=(test_x, test_y), callbacks=[callback])

model.summary()
model.save('SAF.keras')

def prediction(text):
    arr = []
    arr.append(text)
    encoded_text = modelSBERT.encode(arr)
    pre = model.predict(encoded_text)
    predictions = pre[0]

    class_names = multilabel.classes_
    d = {}

    for i in range(0, len(class_names)):
        d[class_names[i]] = predictions[i]

    sorted_predictions = dict(sorted(d.items(), key=lambda x: x[1]))

    for key, value in list(sorted_predictions.items())[-6:]:
        print(key, np.format_float_positional(np.float32(value)))

while True:
    print("\n------------Testing---------------\n")
    text = input("Enter statement to detect context or enter 'E' to exit:").lower()
    if text == "e":
        break
    else:
        prediction(text)
