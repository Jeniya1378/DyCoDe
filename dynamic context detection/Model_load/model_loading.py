import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import os
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow import keras


@keras.utils.register_keras_serializable()
def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1_val = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1_val

def model_load(path=os.path.join(os.getcwd(), 'Model_load\models')):
    model_path = os.path.join(path, 'SAF.keras')
    custom_objects = {
        'f1_score_metric': f1_score_metric
    }
            # 'mean_squared_error': MeanSquaredError(), 
        # 'binary_accuracy': BinaryAccuracy()
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def SBERT_model_load():
    modelBERT = SentenceTransformer('all-MiniLM-L6-v2')
    return modelBERT

classes = ['alarm', 'calculator', 'call', 'camera', 'coffeemachine',
           'direction', 'dress', 'garagedoor', 'greetings', 'information',
           'lights', 'music', 'none', 'notification', 'openapp', 'other',
           'question', 'shopping list', 'shutters', 'stop', 'temperature',
           'time', 'toaster', 'volume', 'weather']

# Load the models once
model = model_load()
modelBERT = SBERT_model_load()

def prediction(text):
    arr = [text]
    encoded_text = modelBERT.encode(arr)
    pre = model.predict(encoded_text)
    predictions = pre[0]

    class_names = classes
    d = {class_names[i]: predictions[i] for i in range(len(class_names))}

    # Sort the predictions dictionary by values in descending order
    sorted_predictions = sorted(d.items(), key=lambda x: x[1], reverse=True)

    # Return the top predicted context and confidence score
    top_context = sorted_predictions[0][0]
    confidence_score = sorted_predictions[0][1]
    return top_context, confidence_score

# # Example usage
# text = "turn on the lights"
# top_context, confidence_score = prediction(text)
# print(f"Top context: {top_context}, Confidence score: {confidence_score}")

# Uncomment the following lines if running interactively
# while True:
#     print("\n------------Testing---------------\n")
#     text = input("Enter statement to detect context or enter 'E' to exit:").lower()
#     if text == "e":
#         break
#     else:
#         top_context, confidence_score = prediction(text)
#         print(f"Top context: {top_context}, Confidence score: {confidence_score}")
