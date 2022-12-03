import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_loss(anchor, positive, negative, alpha):
    """
    The cosine loss is calculated as the sum of the cosine distance between the anchor and positive image, and the anchor and negative image.
    """
    positive_distance = F.cosine_similarity(anchor, positive)
    negative_distance = F.cosine_similarity(anchor, negative)
    
    return torch.sum(torch.max(negative_distance - positive_distance + alpha, torch.zeros_like(positive_distance)))

def batch_triplet_loss(batch_images, batch_recipes, alpha=0.2):
    """
    The loss function takes input as a batch of images and recipes.
    The loss is calculated as the sum of the triplet loss for each image and recipe pair.
    """
    loss = 0
    B = len(batch_images)
    for i, (image, recipe) in enumerate(zip(batch_images, batch_recipes)):
        anchor_image, anchor_recipe = image, recipe
        positive_image, positive_recipe = image, recipe

        # every other image and recipe pair is a negative pair
        loss += cosine_loss(anchor_image, positive_recipe, batch_recipes[range(B != i)], alpha)
        loss += cosine_loss(anchor_recipe, positive_image, batch_images[range(B != i)], alpha)

    return loss

def multi_label_loss(logits, labels):
    """
    Formulate the soft-cross entropy as a multi-label loss function.
    """
    return nn.BCEWithLogitsLoss()(logits, labels)