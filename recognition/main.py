import subprocess

dataset_selection = input("Enter:\n"
                          "1 to select dataset 1\n"
                          "2 to select dataset 2\n"
                          "3 to combine dataset 1 and 2\n"
                          "4 to assign weights to the combined dataset\n"
                          "Your choice: ")

print("\n\n")


model_selection = input("Enter:\n"
                        "1 to run all neural network models\n"
                        "2 to run all neural network models with GloVe embedding\n"
                        "3 to run all neural network models with word2vec embedding\n"
                        "4 to run BERT classifier\n"
                        "5 to run SBERT with FCN(without attention layer)\n"
                        "6 to run SAF(SBERT+Attention module+FCN)\n"
                        "Your choice: ")


if model_selection == "1":
    subprocess.run(['python', 'Deep_neural_networks.py', dataset_selection])
elif model_selection == "2":
    subprocess.run(['python', 'Deep_neural_networks_GloVe.py', dataset_selection])
elif model_selection == "3":
    subprocess.run(['python', 'Deep_neural_networks_word2vec.py', dataset_selection])
elif model_selection == "4":
    subprocess.run(['python', 'BERT.py', dataset_selection])
elif model_selection == "5":
    subprocess.run(['python', 'SBERT+FCN.py', dataset_selection])
elif model_selection == "6":
    subprocess.run(['python', 'SAF4.py', dataset_selection])
else:
    print("Invalid choice. Please enter a valid option (1, 2, 3, 4, 5 or 6).")
