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
import random
import time


# init tree
collection = 'image'
partitions = list_partitions()
load_partitions(partitions)

max_branch_num = 20
max_leaf_size = 1000
walk_multi_branch_threshold = 0.9
structure_path = 'D:/NTU/CSIE/112/DBMS/github/pretrain_database/db-data/cluster_info.pkl'
tree_params = get_user_tree_params(max_branch_num=max_branch_num, 
                                   max_leaf_size=max_leaf_size, 
                                   walk_multi_branch_threshold=walk_multi_branch_threshold, 
                                   structure_path=structure_path)
tree, root = initialize_pretrained_db(**tree_params)

# # search using image and text
# print('========== Test Query ============')
with open('D:/NTU/CSIE/112/DBMS/github/pretrain_database/db-data/save_pairs.pkl', 'rb') as file:
    save_pairs = pickle.load(file)

exp_set_size = 100
seed_value = 21           #6969 #42 #69
random.seed(seed_value)
random_integers = [random.randint(0, len(save_pairs)) for _ in range(exp_set_size)]

save_directory = 'exp_img'
os.makedirs(save_directory, exist_ok=True)

for idx in tqdm.tqdm(random_integers):
    try:
        save_path = os.path.join(save_directory, str(idx)+'.jpg')
        download_image(save_pairs[idx]['img_url'], save_path)
    except Exception:
        pass

retrieval_number = 50

keywords_ours = []
i=0
start_time = time.time()
for idx in tqdm.tqdm(random_integers):
    image_path = os.path.join(save_directory, str(idx)+'.jpg')
    embs, result_info = user_query_preprocess(image_path=image_path, 
                                return_image_url=True,
                                return_image_description=True,
                                return_keywords=True)
    searched_df = retrieve_data_to_user(root, tree, embs, result_info, retrieval_number)
    keywords_ours.append([])
    for j in range(len(searched_df)):
        keywords_ours[i].append(searched_df.iloc[j]['keywords'])
    i+=1
end_time = time.time()
elapsed_time_ours = end_time - start_time



collection = 'exp'

keywords_trad = []
i=0
start_time = time.time()
for idx in tqdm.tqdm(random_integers):
    image_path = os.path.join(save_directory, str(idx)+'.jpg')
    image_embedding = get_clip_embedding(image_path)
    image_embedding = image_embedding.detach().numpy()
    image_embedding = image_embedding.squeeze()
    embedding = np.array(image_embedding)
    res = client.search(
        collection_name=collection,
        data=[embedding],
        limit=retrieval_number,
        output_fields=['keywords']
    )
    keywords_trad.append([])
    for j in range(len(res[0])):
        keywords_trad[i].append(res[0][j]['entity']['keywords'])
    i+=1
    
end_time = time.time()
elapsed_time_trad = end_time - start_time


total_match_ours = 0
total_match_trad = 0
for i in range(len(random_integers)):
    match_ours = 0
    match_trad = 0
    keywords_gt = set(parse_string_to_list(save_pairs[random_integers[i]]['keywords']))
    for j in range(len(keywords_ours[i])):
        for keyword in keywords_ours[i][j]:
            if keyword in keywords_gt:
                match_ours+=1
                break
    for j in range(len(keywords_trad[i])):
        for keyword in keywords_trad[i][j]:
            if keyword in keywords_gt:
                match_trad+=1
                break
    if (match_ours/retrieval_number) >= 0.7:
        total_match_ours += 1
    if (match_trad/retrieval_number) >= 0.7:
        total_match_trad += 1


print('Similarity_threshold:', walk_multi_branch_threshold, '\n')
print('Testing set size:', exp_set_size, '\n')
print('Size of each retrieval:', retrieval_number, '\n')
print(f"Ours elapsed time: {elapsed_time_ours} seconds\n")
print(f"Trad elapsed time: {elapsed_time_trad} seconds\n")
print('Accuracy of ours:', total_match_ours/len(random_integers), '\n')
print('Accuracy of trad:', total_match_trad/len(random_integers), '\n')
    

        

