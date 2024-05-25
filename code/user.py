from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _get_clip_embedding(image_path:str=None, text_input:str=None):
    if image_path and text_input:
        # currently use image info only
        image = Image.open(image_path)
        inputs = processor(text=[text_input], images=image, return_tensors="pt")
    elif image_path and not text_input:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
    elif not image_path and text_input:
        inputs = processor(text=[text], return_tensors="pt")
    else:
        pass
    inputs = inputs.to(device)
    outputs = model(**inputs)
    outputs = outputs.cpu()

    return outputs

def user_query_preprocess(image_path:str=None, 
                          text_input:str=None,
                          return_image_url:bool=True,
                          return_image_description:bool=True,
                          return_keywords:bool=True,
                          ):
    if image_path and text_input:
        # currently use image info only
        image_embedding = _get_clip_embedding(image_path)
        image_embedding = image_embedding.detach().numpy()
        image_embedding = image_embedding.squeeze()
        embedding = np.array(image_embedding)
    elif image_path and not text_input:
        image_embedding = _get_clip_embedding(image_path)
        image_embedding = image_embedding.detach().numpy()
        image_embedding = image_embedding.squeeze()
        embedding = np.array(image_embedding)
    elif not image_path and text_input:
        embedding = _get_clip_embedding(text)
    else:
        print("No Input!!")

    result_info = []
    if return_image_url:
        result_info.append('image_url')
    if return_image_description:
        result_info.append('image_description')
    if return_keywords:
        result_info.append('keywords')

    return embedding, result_info
    
def user_insert_data(image_path:str=None, image_url:str=None, image_description:str=None, keywords:str=None):
    data = {
        'vector':list(_get_clip_embedding(image_path=image_path)),
        'image_url':image_url,
        'image_description':image_description,
        'ai_description':"User",
        'keywords':keywords,
        'group':0,
        'subgroup':-1
    }

    return data

def get_user_tree_params(max_branch_num:int=20, 
                    max_leaf_size:int=2400, 
                    walk_multi_branch_threshold:float=0.6, 
                    structure_path:str=structure_path):
    return {'max_branch_num': max_branch_num,
            'max_leaf_size': max_leaf_size,
            'walk_multi_branch_threshold': walk_multi_branch_threshold,
            'structure_path': structure_path}


