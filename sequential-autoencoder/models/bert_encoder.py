import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import transformers
from tqdm import tqdm

import json
import pickle

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

    def create_embeddings(self, text_inputs):
        """
        Convert raw text input (batches) into embeddings using BERT
        """
        inputs = self.bert_tokenizer(text_inputs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.bert_model(**inputs)
        last_hidden_states = outputs[0]
        
        return last_hidden_states[:, 0, :].detach().cpu().numpy()
    
    def run(self, path_to_vocab, path_to_embeddings):
        """
        Run the BERT encoder to create embeddings for the vocabulary (pickle file)
        """
        embeddings = {}
        with open(path_to_vocab, 'rb') as f:
            vocab = pickle.load(f)
        
        # sort the vocabulary by word
        vocab = sorted(list(vocab.keys()))
        print('Number of words in vocabulary: {}'.format(len(vocab)))
        
        # create batches of words to process from the dictionary
        batch_size = 100
        batches = [vocab[i:i+batch_size] for i in range(0, len(vocab), batch_size)]
        
        # create embeddings for each batch
        print('Creating embeddings...')

        for batch in tqdm(batches):
            embed = self.create_embeddings(batch)

            for word, emb in zip(batch, embed):
                embeddings[word] = emb

        # save the embeddings to a pickle file
        with open(path_to_embeddings, 'wb') as f:
            pickle.dump(embeddings, f)

        print('Embeddings saved to {}'.format(path_to_embeddings))
        
        return