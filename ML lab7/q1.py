import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  silhouette_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
credit_card_data = pd.read_csv('CC GENERAL.csv')  # Replace with your dataset path
numeric_data = credit_card_data.select_dtypes(include=[np.number])
# Preprocess the data 
numeric_data.fillna(0, inplace=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

def k_means_scratch(data, K, max_iterations=100):
   
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    
    for _ in range(max_iterations):
        
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        
       
        for i in range(K):
            cluster_points = data[cluster_assignments == i]
            centroids[i] = np.mean(cluster_points, axis=0)
    
    return cluster_assignments, centroids

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

optimal_K = 4  

cluster_assignments, centroids = k_means_scratch(scaled_data, optimal_K)
kmeans_library = KMeans(n_clusters=optimal_K, n_init=10)  
kmeans_library.fit(scaled_data)
cluster_assignments_library = kmeans_library.labels_
centroids_library = kmeans_library.cluster_centers_

print("Results from Scratch:")
print("Cluster Assignments (First 10):", cluster_assignments[:10])
print("Centroids:", centroids)
print("\nResults from Library Function:")
print("Cluster Assignments (First 10):", cluster_assignments_library[:10])
print("Centroids:", centroids_library)

silhouette_score_scratch = silhouette_score(scaled_data, cluster_assignments)
silhouette_score_library = silhouette_score(scaled_data, cluster_assignments_library)
print("Silhouette Score (Scratch):", silhouette_score_scratch)
print("Silhouette Score (Library):", silhouette_score_library)