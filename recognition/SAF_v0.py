import sys
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Concatenate, Embedding, Attention
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split





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

# Converting to lowercase
df['Voice']=df['Voice'].str.lower()


# Filtering the rare terms.
df = df.groupby("Context",sort=False).filter(lambda x: len(x) > 1)
df.shape

#  Literal evaluate
df['Context'] = df['Context'].apply(lambda x: ast.literal_eval(x))


#  Label Binarization
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Context'])
numOfClasses= len(multilabel.classes_)

# #   Bar graph to show frequencies of classes
# ax = df['Context'].value_counts()[:numOfClasses].plot(kind='barh')
# plt.xlabel("Frequencies")
# plt.ylabel("Contexts")
# plt.show()

y=pd.DataFrame(y, columns=multilabel.classes_)
Y=y[[x for x in multilabel.classes_]]
np.asarray(Y).astype('float32').reshape((-1,1))

#   encoding voice sample to feed in the network
arr = df['Voice'].to_numpy()
modelSBERT = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = modelSBERT.encode(arr)

#   train test split
train1_x, test_x, train1_y, test_y = train_test_split(sentence_embeddings, 
                                                      Y, 
                                                      train_size=0.7, 
                                                      test_size=0.3, 
                                                      random_state=42)


# Attention Layer
# Variable-length int sequences.
query_input = tf.keras.Input(shape=(384,), dtype='float32')
value_input = tf.keras.Input(shape=(384,), dtype='float32')
# Embedding lookup.
token_embedding = tf.keras.layers.Embedding(input_dim=43920, output_dim=384)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=192,
    kernel_size=4,
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)
# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])
# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)
# Concatenate query and document encodings to produce a DNN input layer.
input_layer1 = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])


#Assign weights to contexts
class_frequencies = np.sum(train1_y, axis=0)
total_samples = len(train1_y)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)

class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}


## Creating the layers
input_layer = Input(shape=(384,),tensor=input_layer1)
Layer_1 = Dense(512, activation="relu")(input_layer)
Layer_2 = Dense(256, activation="relu")(Layer_1)
output_layer= Dense(numOfClasses, activation="sigmoid")(Layer_2)
##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer, outputs=output_layer)
## defining the optimiser and loss function
model.compile(
    optimizer='adam',
              loss='mse',metrics=["accuracy","binary_accuracy",tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted')]
)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
## training the model
if dataset_selection == "4":
    trained_model=model.fit(train1_x,train1_y,class_weight=class_weights_dict,epochs=40, batch_size=128,validation_data=(test_x,test_y),callbacks=[callback])
else:
    trained_model=model.fit(train1_x,train1_y,epochs=40, batch_size=128,validation_data=(test_x,test_y),callbacks=[callback])

model.summary()

# def plot_result(item):
#     plt.plot(trained_model.history[item], label=item)
#     plt.plot(trained_model.history["val_" + item], label="val_" + item)
#     plt.xlabel("Epochs")
#     plt.ylabel(item)
#     plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
#     plt.legend()
#     plt.grid()
#     plt.show()


# plot_result("loss")
# plot_result("binary_accuracy")

def prediction(text):
    arr=[]
    arr.append(text)
    encoded_text=modelSBERT.encode(arr)
    pre=model.predict(encoded_text)
    predictions=pre[0]

    class_names=multilabel.classes_
    d={}

    for i in range(0,len(class_names)):
        d[class_names[i]]=predictions[i]
      
    sorted_predictions = dict(sorted(d.items(), key=lambda x: x[1]))

    for key, value in list(sorted_predictions.items())[-6:]:
        print(key, np.format_float_positional(np.float32(value))) 

while(1):
    print("\n------------Testing---------------\n")
    text = input("Enter statement to detect context or enter 'E' to exit:").lower()
    if text == "e":
        break
    else:
        prediction(text)