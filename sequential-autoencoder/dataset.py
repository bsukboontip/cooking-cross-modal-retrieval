import torch
import numpy as np
import torchvision
import torch.functional as F
import os
import sys
import json
import lmdb
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import time
import re
import torchvision.transforms as transforms

class RecipeDataset(torch.utils.data.Dataset):

    """
    Dataset class for loading the title, cleaned ingredients, instructions, list of images, ID for every recipe in the dataset, based on partition (train, validation, test)
    """
    def __init__(self, partition, ids_pkl, cleaned_layers, image_map, dataset_images, bert_embeddings, ingredient_vocabulary, image_logs='', transform=None, seed=42):
        
        self.partition = partition
        self.data = {}
        self.ids = []
        self.image_logs = image_logs
        self.dataset_images = dataset_images
        self.transform = transform
        self.seed = seed
        self.bert_embeddings_path = bert_embeddings
        self.ingredient_vocabulary_path = ingredient_vocabulary

        if self.partition not in ['train', 'val', 'test']:
            raise ValueError('Partition must be one of train, val, test')
        
        with open(cleaned_layers, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} recipes from {cleaned_layers}")
        
        with open(ids_pkl, 'rb') as f:
            self.ids = pickle.load(f)
        # remove bad ids from the ids list
        remove_count = 0
        for id in self.ids:
            if id not in data or data[id]['partition'] != self.partition or data[id]['ingredients'] == [] or data[id]['instructions'] == []:
                self.ids.remove(id)
                remove_count += 1

        print(f"PARTITION: {self.partition}, TOTAL IDS AVAILABLE: {len(self.ids)}")

        # iterate through the data to obtain only samples which are from the partition
        for i, (id, sample) in enumerate(data.items()):
            if sample['partition'] == self.partition:
                self.data[id] = sample
        
        # memory cleanup
        del data

        with open(image_map, 'r') as f:
            self.image_map = json.load(f)
        print(f"Loaded {len(self.image_map)} image mappings from {image_map}")

        torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.random_embedding = torch.randn(768).unsqueeze(0)
        print('random embedding', self.random_embedding.shape)

        with open(self.bert_embeddings_path, 'rb') as f:
            self.bert_embeddings = pickle.load(f)
            self.bert_embeddings = { k: torch.tensor(v).unsqueeze(0) for k, v in self.bert_embeddings.items() }
        with open(self.ingredient_vocabulary_path, 'rb') as f:
            self.ingredient_vocabulary = pickle.load(f)
            
        print(f"Loaded {len(self.bert_embeddings)} embeddings from {self.bert_embeddings_path}\nLoaded {len(self.ingredient_vocabulary['ingredients'])} ingredients from {self.ingredient_vocabulary_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print('INDEX CALLED:', index)
        id = self.ids[index]
        sample = self.data[id]

        image_ids = self.image_map[id]

        # randomly pick out an image from the list of images if train, else pick the first image
        if self.partition == 'train':
            image_id = np.random.choice(image_ids)
        else:
            image_id = image_ids[0]

        # create the image path and load the image
        image_path = self.dataset_images +'/'.join(list(image_id[:4])) + '/' + image_id

        # load image from path
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        except:
            raise ValueError(f"Image not found at path: {image_path}")

        # obtain list of ingredients and instructions
        title = sample['title']
        ingredients = sample['ingredients']
        instructions = sample['instructions']
        print(f"{title}\n{ingredients}\n{instructions}")

        # obtain the embeddings for title, ingredients and instructions from BERT
        # check against the dictionary saved, if not available, then use the random vector generated at the start

        title_embedding = [self.bert_embeddings.get(word, self.random_embedding) for word in title.lower().split(' ')]

        instruction_embedding = []
        for instruction in instructions:
            temp = []
            instruction = re.sub(r"[^a-zA-Z0-9]", " ", instruction.strip().lower())
            # print(f"instruction: {instruction}")
            for word in instruction.split():
                e = self.bert_embeddings.get(word, self.random_embedding)
                temp.append(e)
            
            instruction_embedding.append(torch.cat(temp, dim=0))

        # ingredient embeddings contain an additional lookup in the ingredient vocabulary
        ingredient_embedding = []
        for ingredient in ingredients:
            temp = []
            ingredient = re.sub(r"[^a-zA-Z0-9]", " ", ingredient.strip().lower())
            for word in ingredient.split(" "):
                temp.append(self.bert_embeddings.get(word, self.random_embedding))
            
            ingredient_embedding.append(torch.cat(temp, dim=0))

        # convert the list of embeddings to a tensor, with zero padding to cover variable length
        title_embedding = torch.nn.utils.rnn.pad_sequence(title_embedding, batch_first=True, padding_value=0)
        instruction_embedding = torch.nn.utils.rnn.pad_sequence(instruction_embedding, batch_first=True, padding_value=0)
        ingredient_embedding = torch.nn.utils.rnn.pad_sequence(ingredient_embedding, batch_first=True, padding_value=0)

        # print(f"TITLE EMBEDDING: {torch.squeeze(title_embedding).shape}")
        # print(f"INSTRUCTION EMBEDDING: {instruction_embedding.shape}")
        # print(f"INGREDIENT EMBEDDING: {ingredient_embedding.shape}")
        
        output = {
            'id': id,
            'image_id': image_id,
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions,
            'title_embedding': torch.squeeze(title_embedding),
            'ingredient_embedding': ingredient_embedding,
            'instruction_embedding': instruction_embedding,
            'image': image
        }

        return output

    def visualize_sample(self, index):
        output = self.__getitem__(index)
        print(f"ID: {output['id']}\tImage ID: {output['image_id']}")
        print(f"Title: {output['title']}")
        print(f"Ingredients:")
        for ingredient in output['ingredients']:
            print(f"\t{ingredient}")
        print(f"Instructions:")
        for instruction in output['instructions']:
            print(f"\t{instruction}")

        if self.image_logs:
            image_path = self.dataset_images + '/'.join(list(output['image_id'][:4])) + '/' + output['image_id']
            image_path = os.path.join(self.image_logs, image_path)
            image = Image.open(image_path).convert('RGB')
            image.save(f"{self.image_logs}/{output['id']}.png")

if __name__ == "__main__":
    dataset = RecipeDataset(
        partition='test',
        ids_pkl='/home/ubuntu/recipe-dataset/test/test_keys.pkl', 
        cleaned_layers='/home/ubuntu/recipe-dataset/json/cleaned_layers.json', 
        image_map='/home/ubuntu/recipe-dataset/json/image_map.json', 
        dataset_images='/home/ubuntu/recipe-dataset/test/', 
        bert_embeddings='/home/ubuntu/recipe-dataset/json/vocab_bert.pkl',
        ingredient_vocabulary='/home/ubuntu/recipe-dataset/json/ingredient_vocab.pkl',
        image_logs='/home/ubuntu/cooking-cross-modal-retrieval/sequential-autoencoder/logs',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    dataset.visualize_sample(2)
    print('\n-----------------------------------------------------------\n')
