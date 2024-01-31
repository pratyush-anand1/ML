import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial import distance
from sklearn.metrics import silhouette_score

def simple_DBSCAN(X, clusters, eps, minPts, metric=distance.euclidean):
    currentPoint = 0
    for i in range(0, X.shape[0]):
        if clusters[i] != 0:
            continue
        neighbors = neighborsGen(X, i, eps, metric)
        if len(neighbors) < minPts:
            clusters[i] = -1
        else:
            currentPoint += 1
            expand(X, clusters, i, neighbors, currentPoint, eps, minPts, metric)
    return clusters

def neighborsGen(X, point, eps, metric):
    neighbors = []
    for i in range(X.shape[0]):
        if metric(X[point], X[i]) < eps:
            neighbors.append(i)
    return neighbors

def expand(X, clusters, point, neighbors, currentPoint, eps, minPts, metric):
    clusters[point] = currentPoint
    i = 0
    while i < len(neighbors):
        nextPoint = neighbors[i]
        if clusters[nextPoint] == -1:
            clusters[nextPoint] = currentPoint
        elif clusters[nextPoint] == 0:
            clusters[nextPoint] = currentPoint
            nextNeighbors = neighborsGen(X, nextPoint, eps, metric)
            if len(nextNeighbors) >= minPts:
                neighbors = neighbors + nextNeighbors
        i += 1

class Basic_DBSCAN:
    def __init__(self, eps, minPts, metric=distance.euclidean):
        self.eps = eps
        self.minPts = minPts
        self.metric = metric

    def fit_predict(self, X):
        clusters = [0] * X.shape[0]
        simple_DBSCAN(X, clusters, self.eps, self.minPts, self.metric)
        return clusters

def silhouette_coefficient21(X, labels):
    n = len(X)
    silhouette_values = np.zeros(n)

    for i in range(n):
        a = 0  # Average distance to points in the same cluster
        b = float('inf')  # Average distance to points in the nearest neighboring cluster

        cluster_i = labels[i]

        for j in range(n):
            if i != j:
                if labels[j] == cluster_i:
                    a += distance.euclidean(X[i], X[j])
                else:
                    dist = distance.euclidean(X[i], X[j])
                    if dist < b:
                        b = dist

        a /= max(1, sum(np.array(labels) == cluster_i) - 1)  # Avoid division by zero
        s = (b - a) / max(a, b)
        silhouette_values[i] = s

    return np.mean(silhouette_values)

df = pd.read_csv('housing.csv')
df = df.drop('ocean_proximity', axis=1)
df = df.dropna(axis=0, how='any')
df = df.head(1000)
cols = df.columns

# Perform clustering and get cluster labels
scanner = Basic_DBSCAN(eps=0.85, minPts=7)
X = StandardScaler().fit_transform(df)
clusters = scanner.fit_predict(X)

print(f'Results cluster: {clusters}')

# Calculate silhouette score using the obtained cluster labels
silhouette_avg = silhouette_coefficient21(X, clusters)
print(f"silhoutte coefficient from scratch: {silhouette_avg}")
silhouette_avg1 = silhouette_score(X, clusters)
print(f"Silhouette Coefficient: {silhouette_avg1}")
