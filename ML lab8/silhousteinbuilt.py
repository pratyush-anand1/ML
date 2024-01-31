import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score

df=pd.read_csv('housing.csv')
df=df.drop('ocean_proximity',axis=1)
df = df.dropna(axis=0, how='any') 
df=df.head(1000)
X=StandardScaler().fit_transform(df)
dbsac=DBSCAN(eps=0.8,min_samples=8)
cluster=dbsac.fit_predict(X)
silhouette=silhouette_score(X,cluster)
print(silhouette)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=cluster, palette='bright')
plt.show()
