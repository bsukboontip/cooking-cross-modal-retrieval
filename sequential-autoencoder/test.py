import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast_mode, autocast, GradScaler
import numpy as np
import os
import sys
import time

from models.encoder import CombinedEncoder
from models.image_encoder import ImageEncoder
from models.recipe_encoder import RecipeTransformerEncoder
from models.autoregressive_decoder import LinearDecoder

from loss import multi_label_loss, TripletLoss

# torch.Size([8, 8, 768]) torch.Size([8, 8, 7, 768]) torch.Size([8, 19, 45, 768])

title_embeddings = torch.randn(8, 8, 768).cuda()
ingredient_embeddings = torch.randn(8, 5, 7, 768).cuda()
ingredient_lens = torch.randint(1, ingredient_embeddings.shape[1]+1, (ingredient_embeddings.shape[0],))
instruction_embeddings = torch.randn(8, 19, 45, 768).cuda()
image_embeddings = torch.randn(8, 3, 224, 224).cuda()
batch_triplet_loss = TripletLoss()

encoder = CombinedEncoder().cuda()
decoder = LinearDecoder(vocab_size=9226, input_dim=1024, hidden_dims=[512, 512]).cuda()
print("- TOTAL PARAMETERS:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

# code the forward pass for each timestep in the batch
# for each timestep, we need to compute the loss

# at each step we have incrementally more information from the ingredients
batch_size, num_ingredients, num_ingredient_steps, ingredient_embedding_size = ingredient_embeddings.shape
mask = torch.arange(max(ingredient_lens)).unsqueeze(0) < ingredient_lens.unsqueeze(1)
print(mask)
mask = mask.cuda()
outputs = []
with autocast():
    batch_loss = 0.0
    for t in range(num_ingredients):
        # get the ingredient embeddings until this timestep
        ingredient_embeddings_t = ingredient_embeddings[:, :t+1, :, :]
        print('ingredients at t: ', t, ingredient_embeddings_t.shape)

        image_output, recipe_output = encoder(image_embeddings, title_embeddings, ingredient_embeddings_t, instruction_embeddings)

        print(f"OUTPUTS: {image_output.shape} {recipe_output.shape}")
        batch_loss += batch_triplet_loss(image_output, recipe_output.t())

        # use these to pass through the decoder and get the corresponding outputs
        outputs.append(decoder(torch.cat([image_output, recipe_output], dim=1)))

    outputs = torch.stack(outputs, dim=1)

    # Using this output, compute the average logits across ingredient sequences
    outputs = torch.mean(outputs, dim=1)
    print("OUTPUTS:", outputs.shape)

    labels = torch.randint(0, 2, (batch_size, 9226), dtype=torch.float).cuda()

    label_loss = F.binary_cross_entropy_with_logits(outputs, labels)
    batch_loss = batch_loss / num_ingredients
    print("BATCH LOSS:", batch_loss)
    print("LABEL LOSS:", label_loss)