import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load the Credit Card Customer Data dataset
data = pd.read_csv('Credit_Card_Customer_Data.csv')

# Select the features that you want to use for clustering
X = data[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']].values

# Create an AgglomerativeClustering object
clusterer = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')


# Fit the clusterer to the data
clusterer.fit(X)

# Get the cluster labels for each data point
cluster_labels = clusterer.labels_


# Visualize the clustering results
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
plt.xlabel('Avg_Credit_Limit')
plt.ylabel('Total_Credit_Cards')
plt.title('Agglomerative Hierarchical Clustering')
plt.show()