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
import numpy as npind
from user import *
import time

# # init tree
# max_branch_num = 20
# max_leaf_size = 1000
# walk_multi_branch_threshold = 0.9
# structure_path = 'D:/NTU/CSIE/112/DBMS/github/pretrain_database/db-data/cluster_info.pkl'
# tree_params = get_user_tree_params(max_branch_num=max_branch_num, 
#                                    max_leaf_size=max_leaf_size, 
#                                    walk_multi_branch_threshold=walk_multi_branch_threshold, 
#                                    structure_path=structure_path)
# tree, root = initialize_pretrained_db(**tree_params)




folder_path = 'D:/NTU/CSIE/112/DBMS/github/pretrain_database/OOD-data/food-101-tiny/train/ramen'
valid_extensions = {'.jpg', '.jpeg', '.png'}
image_paths = []

for filename in os.listdir(folder_path):
    _, ext = os.path.splitext(filename)
    if ext.lower() in valid_extensions:
        full_path = os.path.join(folder_path, filename)
        image_paths.append(full_path)
folder_name = os.path.basename(folder_path.rstrip('/\\'))
# folder_name = 'pokemon'

print(len(image_paths))
'''
# for image_path in image_paths:
#     image_url = folder_name
#     image_description = folder_name
#     keywords = [folder_name]
#     try:
#         insert_data = user_insert_data(image_path=image_path, 
#                                     image_url=image_url, 
#                                     image_description=image_description, 
#                                     keywords=keywords)
#         insert_data_to_db(root, tree, insert_data)
#     except Exception:
#         pass
    
    
folder_path = 'D:/NTU/CSIE/112/DBMS/github/pretrain_database/OOD-data/art/engraving'
image_paths = []
for filename in os.listdir(folder_path):
    _, ext = os.path.splitext(filename)
    if ext.lower() in valid_extensions:
        full_path = os.path.join(folder_path, filename)
        image_paths.append(full_path)
folder_name = os.path.basename(folder_path.rstrip('/\\'))
# folder_name = 'pokemon'
random_images = random.sample(image_paths, 50)

retrieval_number = 50
keywords_ours = []
i=0
start_time = time.time()
for image_path in tqdm.tqdm(random_images):
    try:
        embs, result_info = user_query_preprocess(image_path=image_path, 
                                    return_image_url=True,
                                    return_image_description=True,
                                    return_keywords=True)
        searched_df = retrieve_data_to_user(root, tree, embs, result_info, retrieval_number)
        keywords_ours.append([])
        for j in range(len(searched_df)):
            keywords_ours[i].append(searched_df.iloc[j]['keywords'])
        i+=1
    except Exception:
        pass
end_time = time.time()
elapsed_time_ours = end_time - start_time

total_match_ours = 0
for i in range(len(random_images)):
    try:
        match_ours = 0
        keywords_gt = folder_name
        for j in range(len(keywords_ours[i])):
            for keyword in keywords_ours[i][j]:
                if keyword == folder_name:
                    match_ours+=1
                    break
        print(match_ours)
        if (match_ours/retrieval_number) >= 0.5:
            total_match_ours += 1
    except Exception:
        pass
    
print('Accuracy of ours:', total_match_ours/len(random_images), '\n')
'''