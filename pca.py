from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data

# a)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# b
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA - First two principal components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# c)
loss = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_std)
    loss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(K_range, loss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances (Inertia)')
plt.grid(True)
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X_std)




plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')


centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X')

plt.title('PCA and K-Means')
plt.xlabel('First component')
plt.ylabel('Second component')
plt.show()
