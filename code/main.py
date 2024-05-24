from tree import *
import pickle
import random

tree = BPlusTree(max_branch_num=20, max_leaf_size=2400, walk_multi_branch_threshold=0.6)

with open('db-data/cluster_info.pkl', 'rb') as f:
    cluster_info_dict = pickle.load(f)

root = tree.build_pretrained_structure(cluster_info_dict)

leaf = root.children[2].children[0]
data = leaf.load_data()




# new_vector = []
# for i in range(512):
#     new_vector.append(random.random())



# new_data = {'vector': new_vector,
#             'image_url':'566', 
#             'image_description':'54645', 
#             'ai_description':'5465',
#             'keywords':['5456', '8797'],
#             'group':0,
#             'subgroup':-1
# }

# tree.insert(root, new_data)



