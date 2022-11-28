import json
import os
import numpy as np
from tqdm import tqdm
import collections

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


obtain_unique_ingredients(
    '/home/ubuntu/recipe-dataset/json/cleaned_layers.json', 
    '/home/ubuntu/recipe-dataset/json/ingredients.txt'
)