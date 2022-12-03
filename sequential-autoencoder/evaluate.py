import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def evaluate_model(encoder, val_loader, triplet_loss, label_loss, epoch):

    encoder.eval()

    for i, batch in enumerate(val_loader):
        # get the data
        # forward pass
        # compute the loss
        pass