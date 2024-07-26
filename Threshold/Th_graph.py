import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os

def normal_dist(data, color):
    # Create a histogram of the data
    # plt.hist(data, bins=30, density=True, alpha=0.6, color=color, label='Data Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Plot the fitted normal distribution
    plt.plot(x, p, color=color, linewidth=2, label='Normal Distribution')
    plt.yticks([])

    # Add labels and legend
    plt.xlabel('Values')
    # plt.ylabel('Probability Density')
    title = sum(data) / len(data)
    # plt.title('Histogram with Fitted Normal Distribution')
    plt.title(title)
    # plt.legend()
    fig_name = color + '.png'
    plt.savefig(fig_name)

    # Show the plot
    plt.show()

# def plot_graph(x_data, y_data, label):
#     title="Threshold Estimation"
#     plt.plot(x_data, y_data, marker='o', linestyle='-', label=label)
#     plt.xlabel('categories', fontsize=8)
#     plt.ylabel('ranges', fontsize=8)
#     plt.title(title, fontsize=10)
#     plt.legend(fontsize=8)

# Relative file path
file_name = "Threshold_analysis-DCD.xlsx"
file_path = os.path.join(os.path.dirname(__file__), file_name)
sheet_name = '3 categories'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Read data from Excel file using pandas
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    m_data = data[data['Category'] == 'M']['Th']
    n_data = data[data['Category'] == 'N']['Th']
    y_data = data[data['Category'] == 'Y']['Th']

    normal_dist(m_data, 'b')
    normal_dist(n_data, 'r')
    normal_dist(y_data, 'g')
