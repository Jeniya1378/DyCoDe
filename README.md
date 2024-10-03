DyCoDe is a dynamic context detection framework designed for IoT smart homes. It utilizes techniques such as clustering, topic identification, and zero-shot learning to enhance voice assistant capabilities, enabling seamless device control without the need for predefined commands. This project adapts to new appliances and evolving environments, automating tasks with greater accuracy and autonomy. The framework consists of two main pipelines: context recognition and dynamic context detection.

Context Recognition: Utilizes a pre-trained SBERT model with an attention mechanism to identify standard contexts (e.g., lights, temperature) from voice commands. This research work is also available in https://github.com/Jeniya1378/Context-Recognition-from-Voice-Commands-in-Smart-Home

Dynamic Context Detection: When the context recognition model misclassifies data, DyCoDe temporarily stores this data. It then applies clustering algorithms like DBSCAN and OPTICS to group similar voice samples into clusters. A validation process ensures cluster quality, followed by topic identification within each cluster. These topics are mapped to smart home appliances, enabling context-specific actions.

**Getting Started**

To get started with this project, follow these steps:

Clone the Repository: git clone https://github.com/Jeniya1378/DyCoDe

Install Dependencies: Navigate to the project directory and install the necessary dependencies using pip install -r requirements.txt.

Run the System: Please follow the steps provided in file instructions.txt

**Contribution**

We welcome contributions from the community. If you are interested in contributing, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

**License**

Licensed under the MIT License. See the LICENSE file for details.

**Citation**

If you use this code or dataset in your research, please cite:

J. Sultana and R. Iqbal, "A Framework for Context Recognition from Voice Commands and Conversations with Smart Assistants," 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, 2024, pp. 218-221, doi: 10.1109/CCNC51664.2024.10454771.
