import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(x_data, y_data, label):
    title="Comparison of models accuracy(f1 score) for various dataset"
    plt.plot(x_data, y_data, marker='o', linestyle='-', label=label)
    plt.xlabel('Models', fontsize=8)
    plt.ylabel('f1 scores', fontsize=8)
    plt.title(title, fontsize=10)
    plt.legend(fontsize=8)
    

file_path = r'datasets\f1_scores.xlsx'
sheet_name = 'Sheet1'  

# Read data from Excel file using pandas
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract X values
x_values = data['Models']

# Extract Y values for each Y column
y_columns = ['Dataset 1', 'Dataset 2', 'Combined(Imbalanced)', 'Combined(Balanced)']

# Plot each graph
for y_column in y_columns:
    y_values = data[y_column]
    plot_graph(x_values, y_values, label=y_column)

# Adjust the size of the plot and rotate X-axis labels
plt.rcParams["figure.figsize"] = (10, 6)  
plt.xticks(rotation=45, fontsize=6)

plt.savefig("f1_score_SAF.png", dpi=1024, bbox_inches='tight')

# Show all plots
plt.show()

