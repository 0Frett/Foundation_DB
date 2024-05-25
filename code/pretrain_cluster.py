import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

data = np.load('./db-data/embeddings_ALL.npy')
print(f'Data shape: {data.shape}')  # Should print (25000, 512)
sil = []
clus = [5, 10, 15]
for n_clusters in clus:
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    sil.append(silhouette_score(data, labels, metric = 'cosine'))
    tsne = TSNE(n_components=2, random_state=0)
    data_2d = tsne.fit_transform(data)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=sns.color_palette("hsv", n_clusters), legend='full')
    plt.title(f't-SNE visualization of {n_clusters} K-means clusters')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend(title='Cluster')
    plt.savefig(f'./figs/cosine/K-means{n_clusters}_tsne_visualization.png', format='png', dpi=300)
    plt.show()

plt.figure(figsize=(12, 6))
plt.plot(clus,sil)
plt.title('silhoutte score of clusters')
plt.xlabel('clusters')
plt.ylabel('score')
plt.savefig(f'./figs/cosine/sil_cluster.png', format='png', dpi=300)
plt.show()