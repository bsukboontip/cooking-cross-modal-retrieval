import torch
import torch.nn as nn
import numpy as np
import os
import sys


"""
What is this layer doing? 
- Creates a simple parameter tensor, which gets added to the model's parameters, with num_embeddings X hidden_dim as dimensions
- Used as an additional parameter which can be learnt before the encoder is trained
"""
class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer
    Parameters
    ----------
    dropout : float
        Dropout value.
    num_embeddings : int
        Number of embeddings to train.
    hidden_dim : int
        Embedding dimensionality
    """

    def __init__(self, dropout=0.1, num_embeddings=50, embedding_size=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_size))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = embedding_size

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)


"""
The UnitaryTransformerEncoder is a single transformer encoder layer, which is used on a single text embedding
This includes the 
- title
- single ingredient
- single instruction

The text passes through the BERT encoder, and the embedding obtained is passed through this transformer encoder
"""
class UnitaryTransformerEncoder(nn.Module):

    def __init__(self, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1, num_embeddings=50):
        super(UnitaryTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_embeddings = num_embeddings

        # self.embedding = nn.Linear(hidden_dim, hidden_dim)
        self.positional_encoding = LearnedPositionalEncoding(dropout, num_embeddings, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )
        

    def forward(self, x, mask=None):
        # x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # TODO: Check if transpose is needed
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        # TODO: Check if reverse transpose is needed
        x = x.transpose(0, 1)

        x = AvgPoolSequence(torch.logical_not(mask), x)

        return x

class RecipeTransformerEncoder(nn.Module):

    def __init__(self, num_layers=6, num_heads=8, hidden_dim=512, embedding_dim=768):
        super(RecipeTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.title_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.ingredient_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.instruction_embedding = nn.Linear(embedding_dim, hidden_dim)

        # need first level encoders for title, individual ingredients and individual instructions
        self.title_encoder = UnitaryTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim)
        self.ingredient_encoder = UnitaryTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim)
        self.instruction_encoder = UnitaryTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim)

        # need second level encoder for the sequence of ingredients and instructions
        self.ingredient_sequence_encoder = UnitaryTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim)
        self.instruction_sequence_encoder = UnitaryTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim)


    # TODO: Check if mask is required as in the original implementation
    """
    Forward function for the transformer
    params:
        title: list of BERT embedding of the title : [batch_size, seq_len, embedding_dim]
        ingredients: BERT embedding of the ingredients : [batch_size, num_ingredients, seq_len, embedding_dim]
        instructions: BERT embedding of the instructions : [batch_size, num_instructions, seq_len, embedding_dim]
    """
    def forward(self, title, ingredients, instructions):
        batch_size = title.size(0)

        # ------------------------------- Title -------------------------------

        # title : [batch_size, seq_len, embedding_dim]
        title_out = self.title_embedding(title)
        # transpose required for the transformer encoder
        # title : [seq_len, batch_size, hidden_dim]
        mask = (title_out == 0)[:, :, 0]
        title_out = self.title_encoder(title_out, mask=mask)

        # ------------------------------- Ingredients -------------------------------

        # ingredients : [batch_size, num_ingredients, seq_len, embedding_dim]
        # convert to [batch_size * num_ingredients, seq_len, hidden_dim]
        batch_size, num_ingredients, seq_len, embedding_dim = ingredients.size()
        ingredients_out = ingredients.contiguous().view(batch_size*num_ingredients, ingredients.size(2), ingredients.size(3))
        # TODO: mask[:, 0] = 0, why is this required?
        ingredients_out = self.ingredient_embedding(ingredients_out)
        # ingredients : [seq_len, batch_size*num_ingredients, hidden_dim]
        mask = (ingredients_out == 0)[:, :, 0]
        ingredients_out = self.ingredient_encoder(ingredients_out, mask=mask)
        # convert back to [batch_size, num_ingredients, output_dim]
        ingredients_out = ingredients_out.contiguous().view(batch_size, num_ingredients, ingredients_out.size(-1))
        # create new attention mask for the sequence of ingredients
        ingredients_mask = ingredients > 0
        ingredients_mask = (ingredients_mask.sum(-1) > 0).bool()[:, :, 0]
        # pass this through the next level encoder
        ingredients_out = self.ingredient_sequence_encoder(ingredients_out, mask=torch.logical_not(ingredients_mask))

        # ------------------------------- Instructions -------------------------------

        # instructions : [batch_size, num_instructions, seq_len, embedding_dim]
        batch_size, num_instructions, seq_len, embedding_dim = instructions.size()
        instructions_out = instructions.contiguous().view(batch_size*num_instructions, instructions.size(2), instructions.size(3))
        # mask[:, 0] = 0
        instructions_out = self.instruction_embedding(instructions_out)
        # instructions : [batch_size, num_instructions, seq_len, hidden_dim]
        # convert to [batch_size * num_instructions, seq_len, hidden_dim]
        mask = (instructions_out == 0)[:, :, 0]
        instructions_out = self.instruction_encoder(instructions_out, mask=mask)
        # convert back to [batch_size, num_instructions, seq_len, hidden_dim]
        instructions_out = instructions_out.contiguous().view(batch_size, num_instructions, instructions_out.size(-1))
        # create new attention mask for the sequence of instructions
        instructions_mask = instructions > 0
        instructions_mask = (instructions_mask.sum(-1) > 0).bool()[:, :, 0]
        # pass this through the next level encoder
        instructions_out = self.instruction_sequence_encoder(instructions_out, mask=torch.logical_not(instructions_mask))

        return title_out, ingredients_out, instructions_out

# from HT paper
def AvgPoolSequence(attn_mask, feats, e=1e-12):
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats