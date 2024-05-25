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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_user_query(image_file:str, text_inpur:str, )

