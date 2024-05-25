from tree import *
import pickle
import random
import argparse

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_name', type=str, required=True)
#     parser.add_argument('--method', type=str, required=True)
#     parser.add_argument('--text2image', type=str, required=True)
#     parser.add_argument('--img2tab', type=str, required=True)
#     parser.add_argument('--text', type=str, required=True)
#     parser.add_argument('--img', type=str, required=True)
#     parser.add_argument('--tab', type=str, required=True)
#     return parser

def initialize_pretrained_db(max_branch_num:int =20, 
                             max_leaf_size:int =2400, 
                             walk_multi_branch_threshold:int =0.6,
                             structure_path:str =None):

    with open(structure_path, 'rb') as f:
        cluster_info_dict = pickle.load(f)
    tree = BPlusTree(max_branch_num=max_branch_num,
                    max_leaf_size=max_leaf_size,
                    walk_multi_branch_threshold=walk_multi_branch_threshold,)
    root = tree.build_pretrained_structure(cluster_info_dict)
    print('Successfully Build Pretrained DB')

    return tree, root

def insert_data_to_db(root:BPlusTreeNode, tree:BPlusTree, insert_data:dict):
    tree.insert(root, insert_data)
    print('Successfully insert data')

def retrieve_data_to_user(root:BPlusTreeNode, tree:BPlusTree, query_data:dict, return_info:list)
    data_df = tree.retrieve_data(root, query_data, return_info)
    return data_df


if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()

    # user input
    max_branch_num = 5
    max_leaf_size = 1000
    walk_multi_branch_threshold = 0.5
    structure_path = 'db-data/cluster_info.pkl'
    tree, root = initialize_pretrained_db(structure_path=structure_path)

    # test insertion
    insert_vector = [random.random() for i in range(512)]
    insert_data = {'vector': insert_vector,
                'image_url':'566', 
                'image_description':'54645', 
                'ai_description':'5465',
                'keywords':['5456', '8797'],
                'group':0,
                'subgroup':-1
            }
    insert_data_to_db(root, tree, insert_data)

    # test retrieve
    query_vector = [random.random() for i in range(512)]
    return_info = ['image_url', 'image_description']
    insert_data = {'vector': query_vector,
            'image_url':'5679', 
            'image_description':'1321680', 
            'ai_description':'5465',
            'keywords':['5456', '8797'],
            'group':0,
            'subgroup':-1
        }
    retrieve_data_to_user(root, tree, query_data, return_info)

    



