import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys


class AutoRegressiveDecoder(nn.Module):

    def __init__(self, vocab_size, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1, num_embeddings=50):
        super(AutoRegressiveDecoder, self).__init__()

        # adding one for 'unk'... ingredient=0,1,2,3...vocab_size-1, 'unk'=vocab_size
        self.vocab_size = vocab_size + 1
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_embeddings = num_embeddings

        # Input to the transformer decoder is the concatenated image and text embedding
        self.embedding = nn.Linear(hidden_dim*2, hidden_dim)
        # push it back to the hidden dimension
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )

        # output is logits, apply some sort of cross-entropy loss over it
        self.output = nn.Linear(hidden_dim, self.vocab_size)

    """
    Forward pass of the decoder module. 
    
    The encoder creates a memory representation of the image and recipe text predicted until then.
    In the batch, this memory representation has [batch_size, seq_len (varying), hidden_dim] shape.
    Create a corresponding mask for the memory representation, which is used by the decoder to ignore the padding tokens.

    Target sequence is 
    """
    def forward(self, target, memory, target_mask=None, memory_mask=None):
        pass


"""
For the sake of classification, just use a simple MLP after getting the embeddings from the transformer encoder
"""
class LinearDecoder(nn.Module):

    def __init__(self, vocab_size, input_dim=512, hidden_dims: list=[]):
        super(LinearDecoder, self).__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(hidden_dims[-1], vocab_size))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x