import networkx as nx
import plotly.graph_objects as go

def plot_graph_from_file(file_path):
    # Load the graph from the GEXF file
    G = nx.read_gexf(file_path)

    # Use a layout algorithm to position the nodes
    pos = nx.spring_layout(G)

    # Extract edge positions for edges connected to name nodes only
    edge_x = []
    edge_y = []
    for edge in G.edges():
        if G.nodes[edge[0]].get('type') == 'name' or G.nodes[edge[1]].get('type') == 'name':
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Extract node positions for name nodes only
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        if G.nodes[node].get('type') == 'name':
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color='skyblue',
            size=10,
            line_width=2),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Interactive Network Graph of Appliance Names',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Interactive Network Graph using Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)))

    fig.show()

# Path to your GEXF file
gexf_file_path = "graph2.gexf"

# Plot the graph from the file
plot_graph_from_file(gexf_file_path)
