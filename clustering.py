import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('stars_with_gravity.csv')
df.head()

X = df.iloc[:,[3,4]].values
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
  kmeans.fit(X)
  # inertia method returns wcss for that model
  wcss.append(kmeans.inertia_)

plt.plot(range[1,11],wcss)
plt.title("Elbow Method")
plt.xlable("No. Of Clusters")
plt.show()
print(X)