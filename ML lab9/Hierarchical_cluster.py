import numpy as np
import pandas as pd


# Load the dataset from the CSV file
data = pd.read_csv('test2.csv')

# Select the features you want to use for clustering
features = data[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']]

# Convert the selected features to a numpy array
data = features.to_numpy()

# Define a function to compute the distance between two clusters based on the specified linkage
def linkage(linkage_type, cluster1, cluster2):
    if linkage_type == 'single':
        min_distance = float('inf')
        for point1 in cluster1:
            for point2 in cluster2:
                distance = np.linalg.norm(point1 - point2)
                if distance < min_distance:
                    min_distance = distance
        return min_distance
    elif linkage_type == 'complete':
        max_distance = -1
        for point1 in cluster1:
            for point2 in cluster2:
                distance = np.linalg.norm(point1 - point2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance
    elif linkage_type == 'average':
        total_distance = 0
        count = 0
        for point1 in cluster1:
            for point2 in cluster2:
                total_distance += np.linalg.norm(point1 - point2)
                count += 1
        return total_distance / count
    elif linkage_type == 'centroid':
        centroid1 = np.mean(cluster1, axis=0)
        centroid2 = np.mean(cluster2, axis=0)
        return np.linalg.norm(centroid1 - centroid2)

# Define a function to perform agglomerative hierarchical clustering
def hierarchical_clustering(data, linkage_type):
    clusters = [[point] for point in data]

    while len(clusters) > 1:
        min_distance = float('inf')
        merge_indices = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = linkage(linkage_type, clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        del clusters[merge_indices[1]]
        del clusters[merge_indices[0]]
        clusters.append(merged_cluster)

    return clusters[0]

# Perform hierarchical clustering with different linkage functions
linkage_types = ['single', 'complete', 'average', 'centroid']
for linkage_type in linkage_types:
    clusters = hierarchical_clustering(data, linkage_type)
    print(f"Clustering using {linkage_type} linkage:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}:\n", cluster)
    print("\n")
