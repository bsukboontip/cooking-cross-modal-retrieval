import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

# 1. load the list of ingredients and appropriate class labels associated
# 2. train pre-trained BERT classifier
# 3. curate class labels for existing pairs, and save as file

class RecipeIngredientsDataset(Dataset):

    def __init__(self, filename, cuisine_to_idx, pickle_file=None):
        with open(filename, 'r') as f:
            data = json.load(f)

            # load the pickle file and load the labels into a set
            if pickle_file:
                print('STARTED LOADING FILE')
                ids = set(pickle.load(open(pickle_file, 'rb')))
                print('DONE LOADING FILE')
                self.data = [{'recipe_id': k, 'ingredients': v} for k, v in data.items() if k in ids]
            else:
                self.data = [{'recipe_id': k, 'ingredients': v} for k, v in data.items()]

            print(f"TOTAL DATA PROCESSED: {len(self.data)}")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            
            self.cuisine_to_idx = cuisine_to_idx
            self.idx_to_cuisine = {v:k for k,v in self.cuisine_to_idx.items()}

    def __getitem__(self, index):
        ingredients = self.data[index]['ingredients']
        recipe_id = self.data[index]['recipe_id']
        tokenized = self.tokenizer(' '.join(ingredients), padding='max_length', truncation=True)
        tokenized = {key: torch.tensor(val) for key, val in tokenized.items()}

        return tokenized, recipe_id

    def __len__(self):
        return len(self.data)


bert_model = 'distilbert-base-uncased'

class WeakLabelDataset(Dataset):
    
    def __init__(self, filename, partition='train'):
        with open(filename, 'r') as f:
            self.data = json.load(f)
            self.partition = partition
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.cuisine_to_idx = {}

            if self.partition == 'train':
                self.get_class_mapping()

    def get_class_mapping(self):
        cuisines = set()
        for i, item in enumerate(self.data):
            cuisines.add(item['cuisine'])
        cuisines = list(cuisines)
        cuisines.sort()
        print(f"CUISINES: {cuisines}")

        self.cuisine_to_idx = {}
        for i, cuisine in enumerate(cuisines):
            self.cuisine_to_idx[cuisine] = i

    def __len__(self):
        return len(self.data)

    def prepare_data(self, list_of_ingredients):
        return self.tokenizer(' '.join(list_of_ingredients), padding='max_length', truncation=True)

    def __getitem__(self, index):
        
        if self.partition == 'train':
            item = {key: torch.tensor(val) for key, val in self.prepare_data(self.data[index]['ingredients']).items()}
            item['labels'] = self.cuisine_to_idx[self.data[index]['cuisine']]
            
            return item

        else:
            item = {key: torch.tensor(val) for key, val in self.prepare_data(self.data[index]['ingredients']).items()}

            return item

class WeakLabels():

    def __init__(self, train_file, test_file):
        self.train_dataset = WeakLabelDataset(train_file)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def train(self):
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=50, num_workers=4, drop_last=True)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=len(self.train_dataset.cuisine_to_idx))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        num_epochs = 5
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        self.model = self.model.to(self.device)

        progress_bar = tqdm(range(num_training_steps))

        for epoch in tqdm(range(num_epochs)):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
        
        torch.save(self.model.state_dict(), 'model_dict.pt')

    def curate_labels(self):

        self.model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=len(self.train_dataset.cuisine_to_idx))
        self.model.load_state_dict(torch.load('model_dict.pt'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # create a smaller dataset class reading from the cleaned_ingredients JSON file
        # pass it through the same pipeline as the current data loader
        # obtain the outputs for each recipe and map it with the recipe ID
        # save it into a file with the recipe id with label

        target_dataset = RecipeIngredientsDataset(
            '../../recipe1M_layers/cleaned_ingredients.json', 
            self.train_dataset.cuisine_to_idx, 
            pickle_file='../../test/test_keys.pkl')
        target_dataloader = DataLoader(target_dataset, batch_size=200, shuffle=False)
        
        final_output = {}
        for (item, recipe_id) in tqdm(target_dataloader, total=len(target_dataloader)):
            item = {k: v.to(self.device) for k, v in item.items()}
            with torch.no_grad():
                outputs = self.model(**item)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for recipe, prediction in zip(list(recipe_id), predictions.detach().cpu()):
                final_output[recipe] = target_dataset.idx_to_cuisine[int(prediction)]


        json.dump(final_output, open('recipe-cuisine.json', 'w'))



weakLabels = WeakLabels('../../whats-cooking/train.json', '')
# # weakLabels.train()
weakLabels.curate_labels()