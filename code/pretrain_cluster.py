import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score
import time

def get_cluster_num(data):
    #print(f'Data shape: {data.shape}')  # Should print (25000, 512)
    sil = []
    clus = [3, 5, 10, 15, 20]
    for n_clusters in clus:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        sil.append(silhouette_score(data, labels, metric = 'cosine'))
    max_index = sil.index(max(sil))
    cluster_num = clus[max_index]
    return get_cluster_num


def get_subgroup(data, num_clusters, ids):
    time.sleep(3)
    with open('./db-data/emb_info.pkl', 'rb') as pickle_file:
        emb_key = pickle.load(pickle_file)
    with open('./db-data/cluster_info.pkl', 'rb') as pickle_file:
        cluster_info = pickle.load(pickle_file)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    labels = list(kmeans.labels_)
    cluster_centers = kmeans.cluster_centers_
    cluster_info[group_name]['subgroup'] = {
        str(i):{'center':cluster_centers[i]} for i in range(num_clusters)
    }
    assert len(labels) == len(ids)
    for idx in range(len(ids)):
        emb_key[ids[idx]]['subgroup'] = str(labels[idx]) 
    
    with open('./db-data/emb_info.pkl', 'wb') as pickle_file:
        pickle.dump(emb_key, pickle_file)

    with open('./db-data/cluster_info.pkl', 'wb') as pickle_file:
        pickle.dump(cluster_info, pickle_file)
    print(f"success establish subgroup of group{group_name}")
    time.sleep(3)



def recursive_cluster(data, depth, current_depth=0):
    if current_depth >= depth:
        return {'data': data}

    cluster_num = get_cluster_num(data)
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    clusters = {}
    for i in range(cluster_num):
        cluster_data = data[labels == i]
        clusters[i] = recursive_cluster(cluster_data, depth, current_depth+1)
    
    return {
        'labels': labels,
        'clusters': clusters
    }


result = recursive_cluster(data, depth=2)

# The result is a nested dictionary containing cluster labels and sub-clusters
print(result)


if __name__ == '__main__':
    data = np.load('./db-data/embeddings_ALL.npy')
    depth = 2
    

