import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import transformers
from tqdm import tqdm

import json
import pickle
from torch.multiprocessing import Pool, set_start_method

class BERTEncoder():
    """
    BERT Encoder class for converting raw text into embeddings

    Initialize the class with the BERT model and tokenizer
    Create and store a mapping of ingredients/instructions/tokens to their embedding vectors for faster lookup
    """

    def __init__(self, bert_model, bert_tokenizer, device):
        self.bert_model = bert_model.to(device)
        self.bert_tokenizer = bert_tokenizer
        self.device = device
        
        self.ingredient_embeddings = {}
        self.title_embeddings = {}
        self.instruction_embeddings = {}

    def create_embeddings(self, text_input):
        """
        Convert raw text into embeddings using BERT
        """
        input_ids = torch.tensor(self.bert_tokenizer.encode(text_input)).unsqueeze(0).to(self.device)
        outputs = self.bert_model(input_ids)
        last_hidden_states = outputs[0]
        return last_hidden_states[0][0].detach().cpu().numpy()

    def helper_store_embeddings(self, recipe):
        """
        Helper function for storing embeddings from recipe
        Recipe contains title, clebed ingredients, and instructions
        """
        if recipe['title'] not in self.title_embeddings:
            self.title_embeddings[recipe['title']] = self.create_embeddings(recipe['title'])
        for ingredient in recipe['ingredients']:
            if ingredient not in self.ingredient_embeddings:
                self.ingredient_embeddings[ingredient] = self.create_embeddings(ingredient)
        for instruction in recipe['instructions']:
            if instruction not in self.instruction_embeddings:
                self.instruction_embeddings[instruction] = self.create_embeddings(instruction)
    
    def run(self, path_to_recipes):
        """
        Run the BERT encoder on the recipes
        """
        with open(path_to_recipes) as f:
            recipes = json.load(f)
        print("Number of recipes: ", len(recipes))
        
        set_start_method('spawn')
        pool = Pool(2)
        pool.map(self.helper_store_embeddings, recipes.values())
        
        print("Number of ingredient embeddings: ", len(self.ingredient_embeddings))
        print("Number of title embeddings: ", len(self.title_embeddings))
        print("Number of instruction embeddings: ", len(self.instruction_embeddings))

        with open('embeddings/ingredient_embeddings.pkl', 'wb') as f:
            pickle.dump(self.ingredient_embeddings, f)
        with open('embeddings/title_embeddings.pkl', 'wb') as f:
            pickle.dump(self.title_embeddings, f)
        with open('embeddings/instruction_embeddings.pkl', 'wb') as f:
            pickle.dump(self.instruction_embeddings, f)

        return None