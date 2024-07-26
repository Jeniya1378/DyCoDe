import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from dynamic import create_sbert_embeddings
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json


def create_group_of_appliances(appliance_names, appliance_descriptions, graph_file_path):
       # Create a graph
    G = nx.Graph()
    # Add nodes and edges based on similarity

    for i, (appliance1, description1) in enumerate(zip(appliance_names, appliance_descriptions)):
        # print(i)

        G.add_node(appliance1, type='name')  # Adding name node with type 'name'
        G.add_node(description1, type='description')  # Adding description node with type 'description'

        # Connect the name node and description node
        G.add_edge(appliance1, description1, weight=1.0)

        for j, (appliance2, description2) in enumerate(zip(appliance_names, appliance_descriptions)):

            if i != j:

                # Calculate cosine similarity

                # similarity = model.similarity(word, other_word)
                text1_embedding = create_sbert_embeddings(description1, True)
                text2_embedding = create_sbert_embeddings(description2, True)
                similarity = util.cos_sim(text1_embedding, text2_embedding)

                # Only connect nodes with a high similarity

                if similarity > 0.5:

                    G.add_edge(description1, description2, weight=float(similarity))
    
    # Detect communities (synsets)

    print("done with creating edges")

    partition = community_louvain.best_partition(G)
    # Display the communities
    
    # Display the communities

    communities = {}

    for node, comm_id in partition.items():

        if comm_id not in communities:

            communities[comm_id] = [node]

        else:

            communities[comm_id].append(node)
            
            
    print("Detected Synsets (Communities):")

    for comm_id, nodes in communities.items():
        # Filter out description nodes
        name_nodes = [node for node in nodes if G.nodes[node]['type'] == 'name']
        print(f"Community {comm_id}: {name_nodes}")

    

    # Save the graph
    graph_file_path="graph.gexf"
    nx.write_gexf(G, graph_file_path)

    # Save the communities
    community_file_path ="community.json"
    with open(community_file_path, 'w') as f:
        json.dump(partition, f)
 
    return G

def plot_graph(G):
    # Draw the graph
    layout = nx.spring_layout(G)  # Layout algorithm for graph visualization

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Filter out description nodes
    name_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'name']

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, layout, nodelist=name_nodes, node_size=100, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(G, layout, ax=ax)

    # Add labels
    node_labels = {node: node for node in name_nodes}
    nx.draw_networkx_labels(G, layout, labels=node_labels, font_size=5, font_weight="bold", ax=ax)

    # Show the graph
    plt.title("Appliance Similarity Graph")
    plt.show()




def main():
    # Load the Excel file
    excel_file_path = "iot_home_appliances.xlsx"  
    df = pd.read_excel(excel_file_path)
    appliance_names = df["appliances"].tolist()
    appliance_descriptions = df["description"]

    appliance_graph = create_group_of_appliances(appliance_names, appliance_descriptions, 'graph.gexf')
    plot_graph(appliance_graph)



if __name__ == "__main__":
    main()