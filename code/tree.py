import pickle
from sklearn.cluster import KMeans
import numpy as np
from db_access import *


class BPlusTreeNode:
    def __init__(self, name:str=None, embs:list=None, is_leaf:bool=False, parent=None):
        self.is_leaf = is_leaf
        self.standard_embs = embs if embs is not None else []
        self.children = []
        self.size = 0
        self.parent = parent
        self.name = name
        if self.is_leaf:
            create_partition(self.name)

    def load_data(self):
        data_obj = load_data_from_partition(partition=self.name)
        return data_obj
    
    def insert_data(self, data:dict):
        if self.is_leaf == True:
            insert_data_to_partition(partition=self.name, data=data)
              
    def update_size(self):
        if self.is_leaf == True:
            size = get_sizeof_partition(partition=[self.name])
        else:
            size = 0
            for child in self.children:
                size += child.size
        self.size = size
    
    def drop(self):
        drop_partition([self.name])

    def split_child(self):
        # Perform K-means clustering to split the node into two
        data_dict = self.load_data()
        embs = []
        keys = []
        for key, value in data_dict.items():
            embs.append(value['vector'])
            keys.append(key)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(embs)
        labels = list(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        
        # Create two new nodes
        cluster1 = BPlusTreeNode(name=f'{self.name}_0', embs=cluster_centers[0], is_leaf=True, parent=self.parent)
        cluster2 = BPlusTreeNode(name=f'{self.name}_1', embs=cluster_centers[1], is_leaf=True, parent=self.parent)
        
        # Assign embeddings to the new nodes based on the labels
        for idx, label in enumerate(labels):
            if label == 0:
                cluster1.insert_data(cluster1.name, data_dict[keys[idx]])
            else:
                cluster2.insert_data(cluster2.name, data_dict[keys[idx]])
        
        # Update sizes
        cluster1.update_size()
        cluster2.update_size()
        
        # Replace current node with two new nodes
        self.parent = None
        self.drop()

        return cluster1.parent

    def prune_branch(self):
        # Cluster the standard embeddings of children
        # kmeans = KMeans(n_clusters=2, random_state=0)
        # embeddings = np.array([child.standard_embs for child in self.children])
        # kmeans.fit(embeddings)
        # labels = list(kmeans.labels_)
        # cluster_centers = kmeans.cluster_centers_

        # Split children into two groups
        #group1_children = [child for idx, child in enumerate(self.children) if labels[idx] == 0]
        #group2_children = [child for idx, child in enumerate(self.children) if labels[idx] == 1]

        group1_children = []
        group2_children = []
        for idx, child in enumerate(self.children):
            if idx < len(self.children):
                group1_children.append(child)
            else:
                group2_children.append(child)
        
        new_std_emb1 = np.mean([child.emb for child in group1_children])
        new_std_emb2 = np.mean([child.emb for child in group2_children])
        
        # Pull one group to the upper level by creating a new parent node
        new_parent = BPlusTreeNode(name=f'{self.name}_p', embs=new_std_emb1, is_leaf=False, parent=self.parent)
        new_parent.children = group1_children
        
        # Assign the new parent to the children
        for child in group1_children:
            child.parent = new_parent
        
        # Update the current node's children with the other group
        self.children = group2_children
        self.embs = new_std_emb2
        # Update the parent node's children if it's not the root
        if self.parent is not None:
            self.parent.children.append(new_parent)
        else:
            # If the current node is the root, create a new root
            new_root = BPlusTreeNode(name=f'root', embs=None, is_leaf=False, parent=None)
            new_root.children = [self, new_parent]
            self.parent = new_root
            new_parent.parent = new_root
        return new_parent

class BPlusTree:
    def __init__(self, max_branch_num:int, max_leaf_size:int, walk_multi_branch_threshold:float):
        self.max_branch_num = max_branch_num
        self.max_leaf_size = max_leaf_size
        self.walk_multi_branch_threshold = walk_multi_branch_threshold
        self.node_index = []
        self.rightest_index = []

    def build_pretrained_structure(self, pretrained_cluster_info: dict, depth: int = 2):
        def add_children(node, cluster_info, current_depth, target_depth):
            if current_depth < target_depth:
                is_leaf = (current_depth + 1 == target_depth)
                for group_name, group_obj in cluster_info.items():
                    standard_vector = group_obj['center']
                    child_node = BPlusTreeNode(
                        name=f"{group_obj['name']}",
                        embs=list(standard_vector), 
                        is_leaf=is_leaf,
                        parent=node
                        )
                    node.children.append(child_node)
                    if (current_depth+1 < target_depth):
                        add_children(child_node, group_obj['subgroup'], current_depth + 1, target_depth)
        # build tree
        self.root = BPlusTreeNode(name='root', embs=None, is_leaf=False, parent=None)
        add_children(self.root, pretrained_cluster_info, 0, depth)
        return self.root
    
    
    def match_similarity(self, query_vector: list, key_vector: list):
        query_vector = np.array(query_vector)
        key_vector = np.array(key_vector)
        query_norm = query_vector / np.linalg.norm(query_vector)
        key_norm = key_vector / np.linalg.norm(key_vector)
        
        # Compute cosine similarities
        similarity = np.dot(key_norm, query_norm)
        return similarity

    def search(self, node: BPlusTreeNode, query_vector: list, result_nodes: list, result_similarities: list):
        if node.children:
            similarities = []
            max_similarity = -1
            max_node = None
            for i in range(len(node.children)):
                similarity = self.match_similarity(
                    query_vector, node.children[i].standard_embs)
                
                if similarity >= self.walk_multi_branch_threshold:
                    similarities.append(similarity)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_node = node.children[i]
            
            if similarities:
                for i in range(len(similarities)):
                    self.search(
                        node.children[i], query_vector, result_nodes, result_similarities)
            else:
                self.search(max_node, query_vector,
                            result_nodes, result_similarities)
        else:
            result_nodes.append(node)
            result_similarities.append(self.match_similarity(
                query_vector, node.standard_embs))
            


    def insert(self, root: BPlusTreeNode, data:dict):
        candidate_nodes = []
        candidate_similarities = []
        query_vector = data['vector']
        self.search(root, query_vector, candidate_nodes, candidate_similarities)
        result_node = candidate_nodes[np.argmax(candidate_similarities)]
        print(result_node.name)
        #####   Connect to DB and insert data into result_node  #####
        result_node.insert_data(data)

        if result_node.size > self.max_leaf_size:
            parent = result_node.split_child()
            print('split')

        prune_branch_node = result_node.parent
        while prune_branch_node.parent != None:
            if len(prune_branch_node.children) > self.max_branch_num:
                prune_branch_node = prune_branch_node.prune_branch()
            else:
                break

    def merge_similar_semantic_cluster(self):
        pass



