import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast_mode, autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.autoregressive_decoder import LinearDecoder, AutoRegressiveDecoder
from models.encoder import CombinedEncoder
from utils.config import get_args
from dataset import RecipeDataset, collate
from loss import TripletLoss, multi_label_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(args, encoder, decoder, optimizer, train_loader, batch_triplet_loss, label_loss, epoch, scaler):
    
    # train
    encoder.train()
    decoder.train()

    # trackers for loss
    epoch_batch_triplet_loss = 0.0
    epoch_label_loss = 0.0

    for i, batch in enumerate(train_loader):

        # get the data
        title_embeddings = batch['title_embeddings'].to(device)
        ingredient_embeddings = batch['ingredient_embeddings'].to(device)
        ingredient_indexes = batch['ingredient_indexes'].to(device)
        instruction_embeddings = batch['instruction_embeddings'].to(device)
        image_embeddings = batch['image_embeddings'].to(device)
        ingredient_lens = batch['ingredient_lens']
        print(f"INGREDIENT LENGTHS: {ingredient_lens.shape} \n{ingredient_lens}\n")

        batch_size, num_ingredients, num_ingredient_steps, ingredient_embedding_size = ingredient_embeddings.shape
        mask = torch.arange(torch.max(ingredient_lens)).unsqueeze(0) < ingredient_lens.unsqueeze(1)
        mask = mask.to(device)
        
        # --------------------- FORWARD PASS ---------------------
        
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

            # Using this output, compute the average logits across ingredient sequences
            outputs = torch.mean(torch.stack(outputs, dim=1), dim=1)
            print("OUTPUTS:", outputs.shape)

            label_loss = F.binary_cross_entropy_with_logits(outputs, ingredient_indexes)
            batch_loss = batch_loss / num_ingredients
            print("BATCH LOSS:", batch_loss)
            print("LABEL LOSS:", label_loss)

            loss = batch_loss + label_loss

            epoch_batch_triplet_loss += batch_loss.item()
            epoch_label_loss += label_loss.item()

        # --------------------- BACKWARD PASS ---------------------

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --------------------- TRACKING ---------------------
        
        break

    return epoch_batch_triplet_loss, epoch_label_loss


def train():
    args = get_args()

    # set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scaler = GradScaler()

    # load the data
    train_dataset = RecipeDataset(
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # change this later
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    # create the model
    encoder = CombinedEncoder()
    batch_triplet_loss = TripletLoss()
    hidden_dims = [512, 512]
    decoder = LinearDecoder(vocab_size=args.vocab_size, input_dim=args.hidden_dim*2, hidden_dims=hidden_dims)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # create the optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epochs):
        train_one_epoch(args, encoder, decoder, optimizer, train_loader, batch_triplet_loss, multi_label_loss, i, scaler)

        if args.evaluate_freq > 0 and i % args.evaluate_freq == 0:
            # evaluate
            pass

        if args.save_freq > 0 and i % args.save_freq == 0:
            # save the model
            pass

    return
    

train()