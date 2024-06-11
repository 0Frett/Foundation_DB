from functions import *
from pymilvus import MilvusClient
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import pickle
import tqdm
import pandas as pd
import requests
import torch
import numpy as np
from user import *

# init tree
max_branch_num = 10
max_leaf_size = 2000
walk_multi_branch_threshold = 0.5
structure_path = '/db-data/cluster_info.pkl'
tree_params = get_user_tree_params(max_branch_num=max_branch_num, 
                                   max_leaf_size=max_leaf_size, 
                                   walk_multi_branch_threshold=walk_multi_branch_threshold, 
                                   structure_path=structure_path)
tree, root = initialize_pretrained_db(**tree_params)

# # search using image and text
print('========== Test Query ============')
image_path = 'test.png'
text_input = 'A dog on the grass.'
embs, result_info = user_query_preprocess(image_path=image_path, 
                             text_input=text_input,
                             return_image_url=True,
                             return_image_description=True,
                             return_keywords=True)
searched_df = retrieve_data_to_user(root, tree, embs, result_info, 50)
print('data_num', len(searched_df))
print(searched_df.head(5))

print('========== Test Insertion ============')
image_path = 'test.png'
image_url = 'https://hips.hearstapps.com/hmg-prod/images/little-cute-maltipoo-puppy-royalty-free-image-1652926025.jpg?crop=0.444xw:1.00xh;0.129xw,0&resize=980:*'
image_description = 'dog sitting'
keywords = "dog, grass"
insert_data = user_insert_data(image_path=image_path, 
                               image_url=image_url, 
                               image_description=image_description, 
                               keywords=keywords)
insert_data_to_db(root, tree, insert_data)
