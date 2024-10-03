Thank you for being interested to run my project of dynamic context detection from voice commands. 

This project has been built using anaconda and python.

Setup: 

Step 1: Download and install python from https://www.python.org/downloads/

Step 2: Download and install anaconda from https://www.anaconda.com/download

Step 3: Go to anaconda prompt. You can find this from start menu once you have installed anaconda. 

Step 4: 1. Please install the required packages using environment.yml file by running the following commands (3 and 4) in anaconda prompt.
	2. conda env create -f environment.yml
	3. conda activate context_env
	4. pip install -r requirements.txt
	5. You can verify the installation by running command: conda list


Context Recognition:

The SAF model along with other neural network models are provided in the directory "recognition".
Please run the command: python SAF.py
It will train the SAF model with the combined dataset. 
Details about this research work and related paper can be found in https://github.com/Jeniya1378/Context-Recognition-from-Voice-Commands-in-Smart-Home

In order to run the other models for comparison please run the command: python main.py
This will provide options to run -
1. Deep neural network models (CNN, RNN, LSTM, BiLSTM, GRU)
2. Deep neural network with GloVe
3. Deep neural network with word2vec
4. BERT model
5. SBERT+FCN (without attention layer)
6. SAF

Training dataset selection:
1. Public dataset
2. Custom dataset
3. Combined dataset without weight adjustment
4. Combined dataset with weight adjustment

The default value for SAF model is set as 4. Once you run the SAF model it will save the trained model as SAF.keras

Dynamic Context Detection:

Please the command: python context_recognition.py
This utilized the trained SAF.keras model and gives a simple UI to test the model. 
Misclassified commands are saved in the "misclassified_sentence.csv" file.

To run the dynamic context detection approach please run the command: python dynamic.py
It will identify context from the misclassified sentences stored in "misclassified_sentence.csv" file.

For convenience sample misclassifies sentences are provided in file "sample_misclassified_sentence.csv" file. You can either change the file name in dynamic.py file or copy and paste the sentences to "misclassified_sentence.csv" file. 

It will run in loop taking into account each sentence at a time. 
This will take some time to finish execution. 

If you do not want to run in loop rather want to consider the entire file at one go, please comment out the loop execution and intend back the code that is inside the loop. 

Optional:

Threshold Estimation:

The provided excel file contains sample data with related prediction score to analyze the threshold. In order to create graph run the command: python Th_graph.py

Appliance Similarity Graph Creation and Plotting:

The dynamic.py file uses communities of appliances from graph.gexf file. You can create the graph by running the command: python appliance_graph.py

It takes quite long time (30 minutes to several hours based on processing speed). For convenience the graph.gexf file is provided in shared folder.  

To visualize the graph please run the command: python plot_graph.py
This file uses graph file graph2.gexf which is similar to graph.gexf but considers only appliances names for better visualization.

Paraphrasing:

Once the dynamic.py file has identified the contexts from the misclassified sentences, to retrain the SAF model you can first consider augmenting the training samples. To do this please uncomment the code that are at the end of the dynamic.py file main() function. The paraphrasing task utilizes gemini model from google. Please create account in https://ai.google.dev/

use your api key and paste it in gemeni_paraphrase.py file. 

Retraining:

Once training samples has been augmented, you can copy and paste the data in custom dataset and run the command: python SAF.py

