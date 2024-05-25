from pymilvus import MilvusClient
import numpy as np
import json
import pandas as pd
import pickle
import time
from tqdm import tqdm

def parse_string_to_list(input_string):
    """
    Parse a string into several items using a comma as the delimiter
    and save all items in a list.

    Args:
    input_string (str): The string to be parsed.

    Returns:
    list: A list containing the parsed items.
    """
    # Split the input string by the delimiter ','
    items = input_string.split(',')
    # Strip any leading/trailing whitespace from each item
    items = [item.strip() for item in items]

    return items



#########   KEY IN YOUR IP  #########
client = MilvusClient(
    uri="http://192.168.1.111:19530"
)

file_path = "db-data/emb_info.pkl"
with open(file_path, 'rb') as file:
    emb_info = pickle.load(file)
df = pd.read_csv('db-data/test_df.csv')
pair_path = 'db-data/test_save_pairs.pkl' 
s = {}
with open(pair_path, 'wb') as file:
    pickle.dump(s, file)

load = True
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if load:
        with open(pair_path, 'rb') as file:
            save_pairs = pickle.load(file)
        load = False
        time.sleep(3)

    save_pairs[index] = {
        'photo_id' : row['photo_id'],
        'img_url' : row['photo_image_url'],
        'photo_description' : row['photo_description'],
        'ai_description' : row['ai_description'],
        'keywords': row['keywords'],
        'group' : row['group'],
        'subgroup' : row['subgroup'],
        'embeddings' : emb_info[row['photo_id']]['embs']
    }
    if index % 200 == 0:
        with open(pair_path, 'wb') as file:
            pickle.dump(save_pairs, file)
        load = True
        time.sleep(3)

with open(pair_path, 'wb') as file:
    pickle.dump(save_pairs, file)

#pair_path = 'db-data\save_pairs.pkl' 
with open(pair_path, 'rb') as file:
    save_pairs = pickle.load(file)

for key, value in tqdm(save_pairs.items(), total=len(save_pairs)):
    data = {
        'vector':list(value['embeddings']),
        'image_url':str(value['img_url']),
        'image_description':str(value['photo_description']),
        'ai_description':str(value['ai_description']),
        'keywords':parse_string_to_list(value['keywords']),
        'group':value['group'],
        'subgroup':value['subgroup']
    }    
    res = client.insert(
        collection_name="image",
        data=data,
        partition_name=f"{value['group']}_{value['subgroup']}"
    )
