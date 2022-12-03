import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast_mode, autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader


from models.autoregressive_decoder import LinearDecoder, AutoRegressiveDecoder
from models.encoder import CombinedEncoder
from utils.config import get_args
from dataset import RecipeDataset
from loss import batch_triplet_loss, multi_label_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_iteration(args, encoder, decoder, optimizer, train_loader, triplet_loss, label_loss, epoch, scaler):
    
    # train
    encoder.train()
    decoder.train()

    for i, batch in enumerate(train_loader):
        # get the data
        # forward pass
        # compute the loss

        

        pass
    
    return


def train():
    args = get_args()

    # set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scaler = GradScaler()

    # load the data
    train_dataset = RecipeDataset(
        partition='train',
        ids_pkl='/home/ubuntu/recipe-dataset/test/test_keys.pkl', 
        cleaned_layers='/home/ubuntu/recipe-dataset/json/cleaned_layers_trimmed.json', 
        image_map='/home/ubuntu/recipe-dataset/json/image_map.json', 
        dataset_images='/home/ubuntu/recipe-dataset/test/', 
        image_logs='/home/ubuntu/cooking-cross-modal-retrieval/sequential-autoencoder/logs'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # change this later
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create the model
    encoder = CombinedEncoder(*args)
    hidden_dims = [512, 1024, 512]
    decoder = LinearDecoder(vocab_size=args.vocab_size, input_dim=args.embedding_dim, hidden_dims=hidden_dims)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # create the optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epochs):
        train_one_iteration(args, encoder, decoder, optimizer, train_loader, batch_triplet_loss, multi_label_loss, i, scaler)

        if args.evaluate_freq > 0 and i % args.evaluate_freq == 0:
            # evaluate
            pass

        if args.save_freq > 0 and i % args.save_freq == 0:
            # save the model
            pass

    return
    

