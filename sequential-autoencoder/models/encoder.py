import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import ImageEncoder
from recipe_encoder import RecipeTransformerEncoder
from attention import AlternatingCoAttention



"""
Combined Encoder model for encapsulating the image and recipe encoder models
"""
class CombinedEncoder(nn.Module):

    def __init__(self, num_layers=6, num_heads=8, hidden_dim=512, embedding_dim=768, model_type='resnet18', image_embedding_dim=512, pretrained=True):

        super(CombinedEncoder, self).__init__()

        # assign class variables from parameters
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.image_embedding_dim = image_embedding_dim
        self.pretrained = pretrained

        # create image encoder
        self.image_encoder = ImageEncoder(model_type=self.model_type, embedding_dim=self.image_embedding_dim, pretrained=self.pretrained)

        # create recipe encoder
        self.recipe_encoder = RecipeTransformerEncoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim, embedding_dim=self.embedding_dim)

        # create alternating co-attention modules for each of the three encoders
        self.image_title_attention = AlternatingCoAttention(d=self.image_embedding_dim, k=self.embedding_dim)
        self.image_instruction_attention = AlternatingCoAttention(d=self.image_embedding_dim, k=self.embedding_dim)
        self.image_ingredient_attention = AlternatingCoAttention(d=self.image_embedding_dim, k=self.embedding_dim)

        # use a final embedding layer post concatenation to bring output back to embedding_dim
        self.text_embedding_layer = nn.Linear(self.embedding_dim*3, self.embedding_dim)

    
    def forward(self, image, title, instructions, ingredients):
        image_embedding = self.image_encoder(image)
        
        title_embedding, ingredients_embedding, instructions_embedding = self.recipe_encoder(title, ingredients, instructions)

        # apply alternating co-attention to image and recipe embeddings
        title_embedding, image_embedding = self.image_title_attention(title_embedding, image_embedding)
        instructions_embedding, image_embedding = self.image_instruction_attention(instructions_embedding, image_embedding)
        ingredients_embedding, image_embedding = self.image_ingredient_attention(ingredients_embedding, image_embedding)

        # concatenate the three embeddings
        combined_embedding = torch.cat((title_embedding, instructions_embedding, ingredients_embedding), dim=1)
        recipe_embedding = self.text_embedding_layer(combined_embedding)

        return image_embedding, recipe_embedding