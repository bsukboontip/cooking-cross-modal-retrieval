import json
import os
import numpy as np
from tqdm import tqdm
import collections
import torch

from transformers import BertTokenizer, BertModel
from model import BERTEncoder

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def obtain_unique_ingredients(datapath, ingredients_path):
    """
    Obtain the unique ingredients from the dataset
    :param datapath: path to the layers.json file
    :return: list of unique ingredients
    """
    ingredients = collections.defaultdict(int)
    with open(datapath, 'r') as f:
        data = json.load(f)
        for id in tqdm(data, total=len(data)):
            for ingredient in data[id]['ingredients']:
                ingredients[ingredient] += 1
    
    with open(ingredients_path, 'w') as f:
        ingredients = sorted(ingredients, key=ingredients.get, reverse=True)
        for k,v in ingredients.items():
            f.write(k + '\t' + v + '\n')

def get_bert_embeddings():
    """
    Obtain the BERT embeddings for the ingredients
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    bert = BERTEncoder(model, tokenizer, device)

    bert.run('/home/ubuntu/recipe-dataset/json/cleaned_layers.json')

get_bert_embeddings()