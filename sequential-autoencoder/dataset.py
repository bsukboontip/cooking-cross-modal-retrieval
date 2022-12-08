import torch
import numpy as np
import torchvision
import torch.functional as F
import torch.nn.functional as func
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

unk = '<unk>'

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
        self.ids = set(self.ids)
        for i, (id, sample) in enumerate(data.items()):
            if sample['partition'] == self.partition and id in self.ids:
                self.data[id] = sample
        print(f"Loaded {len(self.data)} recipes from {cleaned_layers} for partition {self.partition}")
        # memory cleanup
        del data
        self.ids = list(self.ids)

        with open(image_map, 'r') as f:
            self.image_map = json.load(f)
        print(f"Loaded {len(self.image_map)} image mappings from {image_map}")
        # print("!!! image_map value:", self.image_map['31dee76c25'])

        torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.random_embedding = torch.randn(768).unsqueeze(0)

        with open(self.bert_embeddings_path, 'rb') as f:
            self.bert_embeddings = pickle.load(f)
            self.bert_embeddings = { k: torch.tensor(v).unsqueeze(0) for k, v in self.bert_embeddings.items() }
        with open(self.ingredient_vocabulary_path, 'rb') as f:
            self.ingredient_vocabulary = pickle.load(f)
            
        print(f"Loaded {len(self.bert_embeddings)} embeddings from {self.bert_embeddings_path}\nLoaded {len(self.ingredient_vocabulary['ingredients'])} ingredients from {self.ingredient_vocabulary_path}")
        self.unk_index = len(self.ingredient_vocabulary['stem2ingredient'])
        print(f"UNK index: {self.unk_index}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        sample = self.data[id]

        # print("Indexing with key:", id, "gotten from:", index)
        try:
            image_ids = self.image_map[id]
        except:
            print(f"Image not found with key: {id}")
            return None

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

        # obtain the embeddings for title, ingredients and instructions from BERT
        # check against the dictionary saved, if not available, then use the random vector generated at the start

        title_embedding = [self.bert_embeddings.get(word, self.random_embedding) for word in title.lower().split(' ')]

        instruction_embedding = []
        for instruction in instructions:
            temp = []
            instruction = re.sub(r"[^a-zA-Z0-9]", " ", instruction.strip().lower())
            for word in instruction.split():
                e = self.bert_embeddings.get(word, self.random_embedding)
                temp.append(e)
            if temp:
                instruction_embedding.append(torch.cat(temp, dim=0))

        # ingredient embeddings contain an additional lookup in the ingredient vocabulary
        ingredient_embedding = []
        ingredient_indexes = []
        for ingredient in ingredients:
            temp = []
            stem = self.ingredient_vocabulary['ingredient2stem'].get(ingredient, unk)
            index = self.ingredient_vocabulary['ingredient2index'].get(stem, self.unk_index)
            ingredient_indexes.append(index)
            # if ingredient_indexes[-1] == self.unk_index:
            #     print(f"UNK: {ingredient}")
            ingredient = re.sub(r"[^a-zA-Z0-9]", " ", ingredient.strip().lower())
            for word in ingredient.split(" "):
                temp.append(self.bert_embeddings.get(word, self.random_embedding))
            
            ingredient_embedding.append(torch.cat(temp, dim=0))

        # convert the list of embeddings to a tensor, with zero padding to cover variable length
        title_embedding = torch.nn.utils.rnn.pad_sequence(title_embedding, batch_first=True, padding_value=0)
        instruction_embedding = torch.nn.utils.rnn.pad_sequence(instruction_embedding, batch_first=True, padding_value=0)
        ingredient_embedding = torch.nn.utils.rnn.pad_sequence(ingredient_embedding, batch_first=True, padding_value=0)
        ingredient_indexes = torch.sum(func.one_hot(torch.tensor(ingredient_indexes), num_classes=self.unk_index+1), dim=0, dtype=torch.float)
        
        # print(f"TITLE EMBEDDING: {torch.squeeze(title_embedding).shape}")
        # print(f"INSTRUCTION EMBEDDING: {instruction_embedding.shape}")
        # print(f"INGREDIENT EMBEDDING: {ingredient_embedding.shape}")
        
        output = {
            'id': id,
            'image_id': image_id,
            'title': title,
            'ingredients': ingredients,
            'ingredient_indexes': ingredient_indexes,
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
        print(f"Ingredient Indexes: {output['ingredient_indexes']}")

        if self.image_logs:
            image_path = self.dataset_images + '/'.join(list(output['image_id'][:4])) + '/' + output['image_id']
            image_path = os.path.join(self.image_logs, image_path)
            image = Image.open(image_path).convert('RGB')
            image.save(f"{self.image_logs}/{output['id']}.png")

def collate(batch, need_metadata=False):
    title_embeddings, ingredient_embeddings, instruction_embeddings, images = [], [], [], []
    ingredient_max_seq, ingredient_max_num, instruction_max_seq, instruction_max_num = 0, 0, 0, 0
    ingredient_lens = []
    ingredient_indexes = []

    if need_metadata:
        image_ids = []
        titles = []

    for elem in batch:
        if elem is None:
            continue
        title_embeddings.append(elem['title_embedding'])

        ingredient_max_num = max(ingredient_max_num, elem['ingredient_embedding'].shape[0])
        ingredient_lens.append(elem['ingredient_embedding'].shape[0])
        ingredient_max_seq = max(ingredient_max_seq, elem['ingredient_embedding'].shape[1])
        ingredient_embeddings.append(elem['ingredient_embedding'].unsqueeze(0))
        
        instruction_max_num = max(instruction_max_num, elem['instruction_embedding'].shape[0])
        instruction_max_seq = max(instruction_max_seq, elem['instruction_embedding'].shape[1])
        instruction_embeddings.append(elem['instruction_embedding'].unsqueeze(0))
        
        images.append(elem['image'])

        ingredient_indexes.append(elem['ingredient_indexes'])

        # additional metadata requirement: 'image_id', 'title'
        if need_metadata:
            image_ids.append(elem['image_id'])
            titles.append(elem['title'])
    
    # title
    for i, x in enumerate(title_embeddings):
        if len(x.size()) == 1:
            title_embeddings[i] = x[None, :]
        # print(title_embeddings[i].size())
    title_embeddings = torch.nn.utils.rnn.pad_sequence(title_embeddings, batch_first=True, padding_value=0)
    # print("done for:")
    # for x in title_embeddings:
    #     print(x.size())

    # ingredients
    padded_output_size = np.array([1, ingredient_max_num, ingredient_max_seq, 768])
    for i, elem in enumerate(ingredient_embeddings):
        pad = padded_output_size - np.array(elem.shape)
        ingredient_embeddings[i] = func.pad(elem, (0, pad[3], 0, pad[2], 0, pad[1], 0, pad[0]))
    ingredient_embeddings = torch.cat(ingredient_embeddings, dim=0)
    
    # instructions
    padded_output_size = np.array([1, instruction_max_num, instruction_max_seq, 768])

    for i, elem in enumerate(instruction_embeddings):
        pad = padded_output_size - np.array(elem.shape)
        instruction_embeddings[i] = func.pad(elem, (0, pad[3], 0, pad[2], 0, pad[1], 0, pad[0]))
    instruction_embeddings = torch.cat(instruction_embeddings, dim=0)
    
    # images
    images = torch.stack(images, dim=0)
    ingredient_indexes = torch.stack(ingredient_indexes, dim=0)

    if need_metadata:
        return {
            'title_embeddings': title_embeddings,
            'ingredient_embeddings': ingredient_embeddings,
            'instruction_embeddings': instruction_embeddings,
            'image_embeddings': images,
            'ingredient_lens': torch.tensor(ingredient_lens),
            'ingredient_indexes': ingredient_indexes,
            'image_ids': image_ids,
            'titles': titles
        }    
    return {
        'title_embeddings': title_embeddings,
        'ingredient_embeddings': ingredient_embeddings,
        'instruction_embeddings': instruction_embeddings,
        'image_embeddings': images,
        'ingredient_lens': torch.tensor(ingredient_lens),
        'ingredient_indexes': ingredient_indexes
    }

# if __name__ == "__main__":
#     dataset = RecipeDataset(
#         partition='test',
#         ids_pkl='/home/ubuntu/recipe-dataset/test/test_keys.pkl', 
#         cleaned_layers='/home/ubuntu/recipe-dataset/json/cleaned_layers.json', 
#         image_map='/home/ubuntu/recipe-dataset/json/image_map.json', 
#         dataset_images='/home/ubuntu/recipe-dataset/test/', 
#         bert_embeddings='/home/ubuntu/recipe-dataset/json/vocab_bert.pkl',
#         ingredient_vocabulary='/home/ubuntu/recipe-dataset/json/ingredient_vocab.pkl',
#         image_logs='/home/ubuntu/cooking-cross-modal-retrieval/sequential-autoencoder/logs',
#         transform=transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     )
#     dataset.visualize_sample(0)
