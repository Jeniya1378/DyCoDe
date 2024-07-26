import csv
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_samples
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  # for tokenization
# from nltk.stem import PorterStemmer  # for stemming
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
import json
import networkx as nx
from transformers import pipeline
from gemini_paraphrase import generate_prompt


def get_csv_length():
    file_path = 'misclassified_sentence.csv'
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        length = sum(1 for row in reader)
    return length




# def create_sentence_array(additional_sentences):
#     sentences = []
#     with open('misclassified_sentence.csv', 'r', newline='', encoding='utf-8-sig') as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             sentences.append(row['Sentence'])
    
#     # Extend the sentences list with the additional sentences
#     sentences.extend(additional_sentences)
    
#     # Convert the list of sentences to a NumPy array
#     sentence_array = np.array(sentences)
#     return sentence_array

def create_sentence_array(n):
    sentences = []
    with open('misclassified_sentence.csv', 'r', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            if i >= n:
                break
            sentences.append(row['Sentence'])
    
    # Extend the sentences list with the additional sentences
    # sentences.extend(additional_sentences)
    
    # Convert the list of sentences to a NumPy array
    sentence_array = np.array(sentences)
    return sentence_array

def print_res(cluster_labels, texts):
    # Print the results
    cluster_results = {}

    for i, text in enumerate(texts):
        cluster_label = cluster_labels[i]

        if cluster_label not in cluster_results:
            cluster_results[cluster_label] = []

        cluster_results[cluster_label].append(text)

    # Print the results
    for cluster_label, texts_in_cluster in cluster_results.items():
        print(f"Cluster {cluster_label}:")
        for text in texts_in_cluster:
            print(f"  Text: {text}")
        print()


def silhoute_score(cluster_labels, embeddings):
    # Check if there are more than one unique cluster labels
    if len(set(cluster_labels)) > 1:
        # Calculate silhouette score for each sample in each cluster
        silhouette_scores = silhouette_samples(embeddings, cluster_labels)
        
        # Initialize an empty dictionary to store silhouette scores for each cluster
        scores = {}

        # Calculate and store the average silhouette score for each cluster
        for cluster_label in set(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
            scores[cluster_label] = np.mean(silhouette_scores[cluster_indices])
        return scores
    else:
        # Return an empty dictionary if only one cluster is found
        return {}    
        

def create_sbert_embeddings(texts, bool):
    # Load a pre-trained SBERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    if bool:
            # Compute SBERT embeddings for the text data
        embeddings = sbert_model.encode(texts, convert_to_tensor=True)
    else:
        # Compute SBERT embeddings for the text data
        embeddings = sbert_model.encode(texts)
    return embeddings


def clusters_to_dict(labels, texts, scores):
    clusters_dict = {}
    for label, text in zip(labels, texts):
        if label not in clusters_dict:
            clusters_dict[label] = {'texts': []}
            if label in scores:
                clusters_dict[label]['score'] = float(scores[label])
        clusters_dict[label]['texts'].append(text)
    return clusters_dict

def clustering(embeddings, texts):
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.4, min_samples=2, metric='cosine')  
    dbscan_labels = dbscan.fit_predict(embeddings)

    # Apply OPTICS clustering
    optics_clusterer = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.05)
    optics_labels = optics_clusterer.fit_predict(embeddings)

    dbscan_scores = silhoute_score(dbscan_labels, embeddings)
    optics_scores = silhoute_score(optics_labels, embeddings)

    dbscan_clusters_dict = clusters_to_dict(dbscan_labels, texts, dbscan_scores)
    optics_clusters_dict = clusters_to_dict(optics_labels, texts, optics_scores)

    # print("dbscan :\n", dbscan_clusters_dict)
    # print("optics :\n", optics_clusters_dict)

    return dbscan_clusters_dict, optics_clusters_dict




def filtering_clusters(clusters_dict):
    filtered_clusters_dict = {}
    for label, cluster_data in clusters_dict.items():
        # Check if 'score' key exists and has a valid value
        if 'score' in cluster_data and cluster_data['score'] > 0.2:
            filtered_clusters_dict[label] = cluster_data['texts']
    return filtered_clusters_dict


def creating_similar_clusters(filtered_dbscan, filtered_optics):
    similar_clusters = {}
    # Compare filtered DBSCAN clusters with filtered OPTICS clusters
    for dbscan_label, dbscan_cluster in filtered_dbscan.items():
        for optics_label, optics_cluster in filtered_optics.items():
            # Check if the DBSCAN cluster is a subset of the OPTICS cluster
            if set(dbscan_cluster).issubset(optics_cluster):
                # Store the DBSCAN cluster label and texts in the new dictionary
                if dbscan_label not in similar_clusters:
                    similar_clusters[dbscan_label] = set()  # Use a set to avoid duplicates
                similar_clusters[dbscan_label].update(optics_cluster)
            # Check if the OPTICS cluster is a subset of the DBSCAN cluster
            elif set(optics_cluster).issubset(dbscan_cluster):
                # Store the DBSCAN cluster label and texts in the new dictionary
                if dbscan_label not in similar_clusters:
                    similar_clusters[dbscan_label] = set()  # Use a set to avoid duplicates
                similar_clusters[dbscan_label].update(dbscan_cluster)

    # Convert sets back to lists and print the similar clusters found
    for dbscan_label, similar_data in similar_clusters.items():
        similar_clusters[dbscan_label] = list(similar_data)
        print("DBSCAN Cluster Label:", dbscan_label)
        for text in similar_clusters[dbscan_label]:
            print(text)
        print("---------------------------")

    return similar_clusters



def pre_processing(texts):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenization, stopword removal, and lemmatization
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    filtered_texts = [[word for word in text if word not in stop_words] for text in tokenized_texts]
    lemmatized_texts = [[lemmatizer.lemmatize(word, pos='v') for word in text] for text in filtered_texts]
    return lemmatized_texts


def topic_modeling(clusters):
    # Dictionary to store topics for each cluster
    topics_dict = {}
    # Tokenize, remove stopwords, lemmatize, and apply LDA for each cluster
    for cluster_label, texts_in_cluster in clusters.items():
        lemmatized_texts = pre_processing(texts_in_cluster)
        # Create dictionary
        dictionary = corpora.Dictionary(lemmatized_texts)
        # Convert to bag-of-words corpus and apply LDA
        corpus = [dictionary.doc2bow(text) for text in lemmatized_texts]
        lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
        # Get the highest probable word for each topic
        topics = lda_model.show_topics(num_words=1, formatted=False)
        for topic_id, topic_words in topics:
            highest_prob_word = topic_words[0][0]
            topics_dict[cluster_label] = highest_prob_word
    # print(topics_dict)
    return topics_dict


def find_most_appropriate_community_using_cluster_sentences(G, cluster_sentences, communities):
    # Calculate combined text for each community
    combined_community_texts = {}
    for comm_id, nodes in communities.items():
        if isinstance(nodes, int):  # Check if nodes is an integer
            nodes = [nodes]  # Convert to list

        combined_texts = []
        for node in nodes:
            if node in G.nodes and 'type' in G.nodes[node] and G.nodes[node]['type'] == 'description':
                combined_texts.append(node)
            else:
                print(f"Warning: Node {node} is missing or does not have the 'type' attribute.")

        combined_text = " ".join(combined_texts)
        combined_community_texts[comm_id] = combined_text

    # Calculate combined text embeddings for each community
    combined_community_embeddings = {}
    for comm_id, combined_text in combined_community_texts.items():
        combined_community_embeddings[comm_id] = create_sbert_embeddings(combined_text, True)

    # Calculate similarity with each community for each cluster of sentences
    cluster_community_similarities = {}
    for cluster_label, sentences in cluster_sentences.items():
        cluster_similarity_scores = {}
        for comm_id, community_embedding in combined_community_embeddings.items():
            cluster_similarity_scores[comm_id] = util.cos_sim(create_sbert_embeddings(" ".join(sentences), True), community_embedding).item()
        cluster_community_similarities[cluster_label] = cluster_similarity_scores

    # Find community with highest similarity for each cluster
    most_appropriate_communities = {}
    for cluster_label, similarity_scores in cluster_community_similarities.items():
        most_appropriate_community = max(similarity_scores, key=similarity_scores.get)
        most_appropriate_communities[cluster_label] = most_appropriate_community

    return most_appropriate_communities




def load_community_from_file(community_file_path):
    # Load communities from the JSON file
    with open(community_file_path, 'r') as f:
        communities = json.load(f)
    return communities


def load_graph_from_file(graph_file_path):
    # Load the graph from the GEXF file
    G = nx.read_gexf(graph_file_path)
    return G


def cluster_to_community_print(G, most_appropriate_communities, similar_clusters, communities):
    # Print results
    for cluster_label, community_id in most_appropriate_communities.items():
        print(f"Cluster {cluster_label}:")
        print("Sentences:")
        for sentence in similar_clusters[cluster_label]:
            print(f"- {sentence}")
        print("Appliance Names in Most Appropriate Community:")
        for node in communities[community_id]:
            if G.nodes[node]['type'] == 'name':  # Check if node represents an appliance name
                print(f"- {node}")
        print()


def cluster_topic_to_community_print(G, most_appropriate_communities, topics, communities):
    # Print results
    for topic_label, community_id in most_appropriate_communities.items():
        print(f"Topic {topic_label}:")
        print("Topic Name:", topics[topic_label])
        print("Appliance Names in Most Appropriate Community:")
        for node in communities[community_id]:
            if G.nodes[node]['type'] == 'name':  # Check if node represents an appliance name
                print(f"- {node}")
        print()


# Function to find most appropriate community for a dict of topics
def find_most_appropriate_community_using_topics(topics, communities, G):
    # Calculate similarity with each community for each topic
    topic_community_similarities = {}
    for topic_label, topic_name in topics.items():
        topic_similarity_scores = {}
        for comm_id, nodes in communities.items():
            combined_text = " ".join([node for node in nodes if G.nodes[node]['type'] == 'description'])
            community_embedding = create_sbert_embeddings(combined_text, True)
            topic_embedding = create_sbert_embeddings(topic_name, True)
            similarity = util.cos_sim(topic_embedding, community_embedding).item()
            topic_similarity_scores[comm_id] = similarity
        topic_community_similarities[topic_label] = topic_similarity_scores
    # Find community with highest similarity for each topic
    most_appropriate_communities = {}
    for topic_label, similarity_scores in topic_community_similarities.items():
        most_appropriate_community = max(similarity_scores, key=similarity_scores.get)
        most_appropriate_communities[topic_label] = most_appropriate_community
    
    return most_appropriate_communities


def create_cluster_community_dict(most_appropriate_communities, communities, G):
    # Create a dictionary to store cluster IDs and their corresponding community appliance names
    cluster_community_appliances = {}
    # Populate the dictionary with cluster IDs and their respective community appliance names
    for topic_label, community_id in most_appropriate_communities.items():
        appliance_names = [node for node in communities[community_id] if G.nodes[node]['type'] == 'name']
        cluster_community_appliances[topic_label] = appliance_names


    print(cluster_community_appliances)
    return cluster_community_appliances


# # Function to classify each sentence in a cluster and identify the most appropriate appliance name for each sentence
# def classify_sentences_and_identify_appliance(classifier, cluster_sentences, community_appliance_names):
#     most_appropriate_appliance_names = []
#     for sentence in cluster_sentences:
#         # Classify the sentence using zero-shot classification
#         classification = classifier(sentence, community_appliance_names)
#         # Extract the appliance name with the highest score
#         most_appropriate_appliance_name = classification['labels'][0]
#         # Append the most appropriate appliance name to the list
#         most_appropriate_appliance_names.append(most_appropriate_appliance_name)
#     return most_appropriate_appliance_names


# # Function to classify sentences in each cluster and identify appliance names for each cluster
# def classify_clusters_and_identify_appliance(classifier, cluster_sentences_dict, community_appliance_names_dict):
#     results = {}
#     for cluster_label, cluster_sentences in cluster_sentences_dict.items():
#         community_appliance_names = community_appliance_names_dict[cluster_label]
#         result = classify_sentences_and_identify_appliance(classifier, cluster_sentences, community_appliance_names)
#         results[cluster_label] = result
#     return results


# Function to classify each sentence in a cluster and identify the most appropriate appliance name with scores
def classify_sentences_and_identify_appliance(classifier, cluster_sentences, community_appliance_names):
    classified_sentences = []
    for sentence in cluster_sentences:
        # Classify the sentence using zero-shot classification
        classification = classifier(sentence, community_appliance_names)
        # Extract the appliance name with the highest score and its score
        most_appropriate_appliance_name = classification['labels'][0]
        prediction_score = classification['scores'][0]
        # Append the most appropriate appliance name and score to the list
        classified_sentences.append((sentence, most_appropriate_appliance_name, prediction_score))
        # classified_sentences.append((most_appropriate_appliance_name, prediction_score))
    return classified_sentences

# Function to classify sentences in each cluster and identify appliance names for each cluster with scores
def classify_clusters_and_identify_appliance(classifier, cluster_sentences_dict, community_appliance_names_dict):
    results = {}
    for cluster_label, cluster_sentences in cluster_sentences_dict.items():
        community_appliance_names = community_appliance_names_dict[cluster_label]
        result = classify_sentences_and_identify_appliance(classifier, cluster_sentences, community_appliance_names)
        results[cluster_label] = result
    return results


def save_results_to_dict(results, cluster_sentences_dict):
    result_dict = {}
    sentences_to_remove = []
    for cluster_label, result in results.items():
        cluster_sentences = cluster_sentences_dict[cluster_label]  # Get the sentences for the current cluster
        for sentence, appliance_name in zip(cluster_sentences, result):
            if appliance_name not in result_dict:
                result_dict[appliance_name] = []
            result_dict[appliance_name].append(sentence)
            sentences_to_remove.append(sentence)
    remove_sentences_from_csv('misclassified_sentence.csv', sentences_to_remove)
    return result_dict


def remove_sentences_from_csv(file_path, sentences_to_remove):
    df = pd.read_csv(file_path)
    df = df[~df['Sentence'].isin(sentences_to_remove)]
    df.to_csv(file_path, index=False)


def load_results_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            result_dict = json.load(file)
    except FileNotFoundError:
        result_dict = {}
    return result_dict

def save_results_to_file(result_dict, file_path):
    with open(file_path, 'w') as file:
        json.dump(result_dict, file, indent=4)

def check_and_save_results(result_dict, results_file_path, threshold=10):
    keys_to_save = [key for key, values in result_dict.items() if len(values) >= threshold]
    
    if keys_to_save:
        # Create a dictionary with only the keys that meet the threshold
        save_dict = {key: result_dict[key] for key in keys_to_save}
        
        # Save this dictionary to a file
        save_results_to_file(save_dict, results_file_path)
        
        # Remove saved entries from the main result_dict
        for key in keys_to_save:
            del result_dict[key]
    
    return result_dict

# def print_final(results, cluster_sentences_dict):
#     for cluster_label, result in results.items():
#         print(f"Cluster {cluster_label}:")
#         cluster_sentences = cluster_sentences_dict[cluster_label]  # Get the sentences for the current cluster
#         for i, (sentence, appliance_name) in enumerate(zip(cluster_sentences, result)):
#             print(f"Sentence {i+1}:")
#             print(f"- Text: {sentence}")
#             print(f"  Appliance Name: {appliance_name}")

def update_results_dict(results, cluster_sentences_dict, result_dict):
    sentences_to_remove = []
    for cluster_label, result in results.items():
        cluster_sentences = cluster_sentences_dict[cluster_label]  # Get the sentences for the current cluster
        for sentence, appliance_name in zip(cluster_sentences, result):
            if appliance_name not in result_dict:
                result_dict[appliance_name] = []
            result_dict[appliance_name].append(sentence)
            sentences_to_remove.append(sentence)
    # remove_sentences_from_csv('misclassified_sentence.csv', sentences_to_remove)
    return result_dict

def log_full_classes(result_dict, threshold=10):
    for key, values in result_dict.items():
        if len(values) >= threshold:
            print(f"Class '{key}' has {len(values)} sentences and meets the threshold of {threshold}.")



def get_high_score_sentences(result_dict, threshold=0.50):
    appliance_sentences = {}

    for cluster_label, cluster_data in result_dict.items():
        for sentence, appliance_name, score in cluster_data:
            if score >= threshold:
                if appliance_name not in appliance_sentences:
                    appliance_sentences[appliance_name] = []
                appliance_sentences[appliance_name].append(sentence)
    return appliance_sentences


def calculate_paraphrases_needed(appliance_sentences, target_count=10):
    paraphrase_counts = {}

    for appliance, sentences in appliance_sentences.items():
        existing_count = len(sentences)
        if existing_count < target_count:
            paraphrases_needed = target_count - existing_count
            paraphrases_per_sentence = paraphrases_needed // existing_count
            extra_paraphrases = paraphrases_needed % existing_count
            paraphrase_counts[appliance] = {
                "total_needed": paraphrases_needed,
                "per_sentence": paraphrases_per_sentence,
                "extra_needed": extra_paraphrases
            }
        else:
            paraphrase_counts[appliance] = {
                "total_needed": 0,
                "per_sentence": 0,
                "extra_needed": 0
            }

    return paraphrase_counts

def retraining_check(result_dict):
    # Get sentences with high scores
    retraining_samples = get_high_score_sentences(result_dict)
    print("High score sentences:", retraining_samples)
    
    # Calculate the number of paraphrases needed
    paraphrase_counts = calculate_paraphrases_needed(retraining_samples)
    print("Paraphrase counts:", paraphrase_counts)
    
    # Dictionary to store paraphrased texts with appliance names
    paraphrased_texts = {}
    
    # Generate and store the paraphrased text
    for appliance, sentences in retraining_samples.items():
        paraphrase_count_info = paraphrase_counts[appliance]
        total_needed = paraphrase_count_info['total_needed']
        per_sentence = paraphrase_count_info['per_sentence']
        
        paraphrased_texts[appliance] = []
        
        for sentence in sentences:
            # Generate paraphrases for each sentence
            for _ in range(per_sentence):
                paraphrased_text = generate_prompt(sentence, 1)  # Generate one paraphrase at a time
                paraphrased_texts[appliance].append(paraphrased_text)
                
            # If there are still more paraphrases needed after equal distribution
            while len(paraphrased_texts[appliance]) < total_needed + len(sentences):
                paraphrased_text = generate_prompt(sentence, 1)
                paraphrased_texts[appliance].append(paraphrased_text)
    
    # Print the paraphrased texts for each appliance
    for appliance, texts in paraphrased_texts.items():
        print(f"Paraphrased texts for {appliance}:")
        for text in texts:
            print(text)
        print("---------------------------")
    
    return paraphrased_texts





def main():
    # initial_texts = []
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Initialize the results dictionary
    result_dict = {}
    length = get_csv_length()
    print(length)

    for n in range(2, length):
        # Call the function to create the DataFrame
        # sentence_df = create_sentence_array(initial_texts, n)
        sentence_df = create_sentence_array(n)
        print(len(sentence_df))
        sentence_embeddings = create_sbert_embeddings(sentence_df, False)
        dbscan_clusters_dict, optics_clusters_dict = clustering(sentence_embeddings, sentence_df)

        filtered_dbscan = filtering_clusters(dbscan_clusters_dict)
        filtered_optics = filtering_clusters(optics_clusters_dict)

        clusters = creating_similar_clusters(filtered_dbscan, filtered_optics)
        print(clusters)

        topics = topic_modeling(clusters)
        print(topics)

        community_file_path = "community.json"
        graph_file_path = "graph.gexf"

        communities = load_community_from_file(community_file_path)
        G = load_graph_from_file(graph_file_path)
        # most_appropriate_communities_using_cluster = find_most_appropriate_community_using_cluster_sentences(G, clusters, communities)
        most_appropriate_communities_using_topics = find_most_appropriate_community_using_topics(topics, communities, G)

        cluster_community_appliances = create_cluster_community_dict(most_appropriate_communities_using_topics, communities, G)

        # Define the zero-shot classification pipeline
        classifier = pipeline("zero-shot-classification")

        results = classify_clusters_and_identify_appliance(classifier, clusters, cluster_community_appliances)

        # training_data = retraining_check(results)

        # Update the results in the dictionary
        # updated_result_dict = update_results_dict(results, clusters, result_dict)
        
        # Check and log if any key has 10 or more values
        # log_full_classes(updated_result_dict, threshold=10)

        # print(updated_result_dict)
        print(results)

        # print(training_data)




if __name__ == "__main__":
    main()
