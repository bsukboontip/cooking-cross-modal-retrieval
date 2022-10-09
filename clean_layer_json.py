import json
import os
import sys
from tqdm import tqdm
import time


def clean_layer_1(filename, cleaned_ingredients_file):
    cleaned = {}

    start = time.time()
    layer1 = json.load(open(filename, 'r'))
    ingredients = json.load(open(cleaned_ingredients_file, 'r'))
    end = time.time()
    print(f"File Loading: {end-start}")

    for recipe in tqdm(layer1):
        cleaned[recipe['id']] = {
            'partition': recipe['partition'],
            'ingredients': ingredients[recipe['id']],
            'instructions': [item['text'] for item in recipe['instructions']],
            'title': recipe['title'],
            'url': recipe['url'],
        }

    json.dump(cleaned, open('cleaned_layers.json', 'w'))

def clean_ingredients(ingredients_file):
    ingredients = json.load(open(ingredients_file, 'r'))
    cleaned = {}
    for recipe in tqdm(ingredients):
        ingredients = [item['text'] for (val, item) in zip(recipe['valid'], recipe['ingredients']) if val]
        cleaned[recipe['id']] = ingredients
    
    json.dump(cleaned, open('cleaned_ingredients.json', 'w'))



clean_layer_1('../recipe1M_layers/layer1.json', '../recipe1M_layers/cleaned_ingredients.json')

# clean_ingredients('../det_ingrs.json')
