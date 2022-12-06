import json
import os
import numpy as np
from tqdm import tqdm
import collections
import torch
import pickle
import string

from transformers import BertTokenizer, BertModel
from models.bert_encoder import BERTEncoder

from nltk.stem import PorterStemmer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def obtain_recipe_vocabulary(path_to_recipe, path_to_vocab):
    """
    Obtain the vocabulary of the recipes
    :param path_to_recipe: path to the layers.json file
    :param path_to_vocab: path to the vocabulary file
    """
    from nltk.corpus import stopwords
    vocab = collections.defaultdict(int)
    stopwords = set(stopwords.words('english'))

    with open(path_to_recipe, 'r') as f:
        data = json.load(f)
        for id in tqdm(data, total=len(data)):
            recipe = data[id]
            
            # split title, ingredients and instructions and add to vocabulary
            for word in recipe['title'].split():
                word = word.translate(str.maketrans('', '', string.punctuation)).lower()
                if word not in stopwords:
                    vocab[word] += 1
            for instruction in recipe['instructions']:
                for word in instruction.split():
                    word = word.translate(str.maketrans('', '', string.punctuation)).lower()
                    if word not in stopwords:
                        vocab[word] += 1
            for ingredient in recipe['ingredients']:
                for word in ingredient.split():
                    word = word.translate(str.maketrans('', '', string.punctuation)).lower()
                    if word not in stopwords:
                        vocab[word] += 1

    # clean up the vocabulary to remove words that appear less than 10 times
    vocab = {k:v for k,v in vocab.items() if v >= 10}

    # save the vocabulary to a pickle file
    with open(path_to_vocab, 'wb') as f:
        pickle.dump(vocab, f)

def trim_dataset(path_to_recipe, path_to_new_recipe):
    """
    Trim the dataset to only contain recipes with at least 3 ingredients
    :param path_to_recipe: path to the layers.json file
    """
    temp = {}
    with open(path_to_recipe, 'r') as f:
        data = json.load(f)
        for id in tqdm(data, total=len(data)):
            if np.random.random() < 0.1:
                temp[id] = data[id]
    
    with open(path_to_new_recipe, 'w') as f:
        json.dump(temp, f)

def obtain_unique_ingredients(path_to_recipe, path_to_ingredient_vocab):
    """
    Obtain the unique ingredients from the dataset
    :param path_to_recipe: path to the layers.json file
    :param path_to_ingredient_vocab: path to the ingredient vocabulary file
    """
    ps = PorterStemmer()
    ingredients = collections.defaultdict(int)
    stem2ingredient = collections.defaultdict(list)
    ingredient2stem = collections.defaultdict(str)
    with open(path_to_recipe, 'r') as f:
        data = json.load(f)
        for id in tqdm(data, total=len(data)):
            for ingredient in data[id]:
                stem = ps.stem(ingredient)
                
                ingredients[stem] += 1
                stem2ingredient[stem].append(ingredient)
                ingredient2stem[ingredient] = stem
    
    # prune the ingredients to only contain ingredients that appear more than 10 times
    ingredients = {k:v for k,v in ingredients.items() if v >= 10}

    # create a mapping from ingredient to index, and index to ingredient
    ingredient2index = {ingredient: i for i, ingredient in enumerate(ingredients)}
    index2ingredient = {v: k for k, v in ingredient2index.items()}
    stem2ingredient = {k:v for k,v in stem2ingredient.items() if k in ingredients}
    ingredient2stem = {k:v for k,v in ingredient2stem.items() if v in ingredients}

    # save the ingredients to a pickle file
    with open(path_to_ingredient_vocab, 'wb') as f:
        obj = {
            'ingredients': ingredients,
            'stem2ingredient': stem2ingredient,
            'ingredient2stem': ingredient2stem,
            'ingredient2index': ingredient2index,
            'index2ingredient': index2ingredient
        }
        pickle.dump(obj, f)

obtain_unique_ingredients('/home/ubuntu/recipe-dataset/json/cleaned_ingredients.json', '/home/ubuntu/recipe-dataset/json/ingredient_vocab.pkl')