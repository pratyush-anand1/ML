import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv('cccd.csv')

# Select the features you want to use for clustering
features = data[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# Number of clusters
n_clusters = 4  # You can change the number of clusters as needed

# Create an AgglomerativeClustering instance with the desired linkage type
linkage_types = ['single', 'complete', 'average', 'centroid']  # You can choose from these linkage types

for linkage_type in linkage_types:
    # Fit the AgglomerativeClustering model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    model.fit(data_scaled)

    # Add cluster labels to the original dataset
    data['Cluster_Labels'] = model.labels_

    # Print and visualize the results
    print(f"Clustering using {linkage_type} linkage:")
    print(data.groupby('Cluster_Labels').mean())

    # You can also create a scatter plot to visualize the clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Avg_Credit_Limit'], data['Total_Credit_Cards'], c=data['Cluster_Labels'], cmap='rainbow')
    plt.xlabel('Avg_Credit_Limit')
    plt.ylabel('Total_Credit_Cards')
    plt.title(f"Clustering using {linkage_type} linkage")
    plt.show()
