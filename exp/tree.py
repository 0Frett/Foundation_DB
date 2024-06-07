import pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from db_access import *
import torch


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
        
    def bfs(self, root):
        if not root:
            return []
        queue = [(root, None)]  # Queue of tuples (node, layer, parent name)
        sequence = []

        while queue:
            current_node, parent_name = queue.pop(0)
            sequence.append((current_node.name, parent_name))
            
            # Enqueue all children of the current node with incremented layer and current node's name as parent name
            for child in current_node.children:
                queue.append((child, current_node.name))
        
        return sequence
            

    def update_size(self):
        if self.is_leaf == True:
            size = get_sizeof_partition(partition=[self.name])
        else:
            size = 0
            for child in self.children:
                size += child.size
        self.size = size
    
    def drop(self):
        drop_partition(self.name)

    def load_data(self):
        data_obj = load_data_from_partition(partition=self.name)
        return data_obj
    
    def insert_data(self, data:dict):
        if self.is_leaf == True:
            insert_data_to_partition(partition=self.name, data=data)
            self.update_size()
        else:
            print('Cannot insert into non-leaf node')

    def split_child(self):
        # Perform K-means clustering to split the node into two
        data_dict = self.load_data()
        embs = []

        for dict in data_dict:
            embs.append(dict['vector'])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(embs)
        labels = list(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        
        # Create two new nodes
        cluster1 = BPlusTreeNode(name=f'{self.name}_0', embs=cluster_centers[0], is_leaf=True, parent=self.parent)
        cluster2 = BPlusTreeNode(name=f'{self.name}_1', embs=cluster_centers[1], is_leaf=True, parent=self.parent)
        
        # Assign embeddings to the new nodes based on the labels
        
        for idx, label in enumerate(labels):
            if label == 0:
                del data_dict[idx]['id']
                cluster1.insert_data(data_dict[idx])
            else:
                del data_dict[idx]['id']
                cluster2.insert_data(data_dict[idx])
            
        # Update sizes
        cluster1.update_size()
        cluster2.update_size()
        
        # Replace current node with two new nodes
        self.drop()
        parent = self.parent
        parent.children.remove(self)
        parent.children.append(cluster1)
        parent.children.append(cluster2)
        self.parent = None
        
        return cluster1.parent

    def prune_branch(self):
        # Cluster the standard embeddings of children
        kmeans = KMeans(n_clusters=2, random_state=0)
        embeddings = np.array([child.standard_embs for child in self.children])
        kmeans.fit(embeddings)
        labels = list(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_

        # Split children into two groups
        group1_children = [child for idx, child in enumerate(self.children) if labels[idx] == 0]
        group2_children = [child for idx, child in enumerate(self.children) if labels[idx] == 1]

        # new_std_emb1 = np.mean([child.emb for child in group1_children])
        # new_std_emb2 = np.mean([child.emb for child in group2_children])
        
        # Pull one group to the upper level by creating a new parent node
        new_parent = BPlusTreeNode(name=f'{self.name}_p', embs=cluster_centers[0], is_leaf=False, parent=self.parent)
        new_parent.children = group1_children
        
        # Assign the new parent to the children
        for child in group1_children:
            child.parent = new_parent
        
        # Update the current node's children with the other group
        self.children = group2_children
        self.embs = cluster_centers[1]
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
    
    def match_similarity(self, query_vector, key_vector: list):
        query_vector = np.array(query_vector)
        key_vector = np.array(key_vector)
        query_norm = query_vector / np.linalg.norm(query_vector)
        key_norm = key_vector / np.linalg.norm(key_vector)
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
        print(f'insert data into leaf node {result_node.name}')
        #####   Connect to DB and insert data into result_node  #####
        result_node.insert_data(data)
        
        prune_branch_node = result_node.parent
        
        if result_node.size > self.max_leaf_size:
            parent = result_node.split_child()
            print(f'split node {result_node.name}')
            # print(root.bfs(root))

        while prune_branch_node.parent != None:
            # print(len(prune_branch_node.children), self.max_branch_num)
            if len(prune_branch_node.children) > self.max_branch_num:
                print(f'prune node {prune_branch_node.name} branches')
                prune_branch_node  = prune_branch_node.prune_branch()
                # print(root.bfs(root))
            else:
                break
        
    
    def retrieve_data(self, root: BPlusTreeNode, query_vector: list, return_info:list, total_number: int):
        candidate_nodes = []
        candidate_similarities = []
        self.search(root, query_vector, candidate_nodes, candidate_similarities)
        # print(f'retrieve data from node(s): {[node.name for node in candidate_nodes]}')
        datas_to_return = []
        N = int(total_number / len(candidate_nodes))
        if N <= 0:
            N = 1
        for node in candidate_nodes:
            result = similarity_search(node.name, query_vector, N)
            ids = get_ids_by_similarity_search_result(result)
            data_dict = get_entities_by_ids(node.name, ids)
            for dict in data_dict:
                row = {}
                for key, value in dict.items():
                    if key in return_info:
                        row[f'{key}'] = value
                datas_to_return.append(row)
        return_df = pd.DataFrame(datas_to_return)
        return return_df
                
    def merge_similar_semantic_cluster(self):
        pass



