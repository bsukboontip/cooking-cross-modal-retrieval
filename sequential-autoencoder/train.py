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
import gc
from evaluate import run_evaluation
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(args, encoder, decoder, optimizer, train_loader, batch_triplet_loss, label_loss, epoch, scaler, count):
    
    # train
    encoder.train()
    decoder.train()

    # trackers for loss
    epoch_batch_triplet_loss = 0.0
    epoch_label_loss = 0.0

    for i, batch in tqdm(enumerate(train_loader), total=4000):
        if i == 4000:
            return epoch_batch_triplet_loss, epoch_label_loss, count
        
        count += 1

        # get the data
        with autocast(dtype=torch.float16):
            title_embeddings = batch['title_embeddings'].type(torch.float16).to(device)
            ingredient_embeddings = batch['ingredient_embeddings'].type(torch.float16).to(device)
            ingredient_indexes = batch['ingredient_indexes'].type(torch.float16).to(device)
            instruction_embeddings = batch['instruction_embeddings'].type(torch.float16).to(device)

            image_embeddings = batch['image_embeddings'].type(torch.float16).to(device)
            ingredient_lens = batch['ingredient_lens'].type(torch.int8)

            batch_size, num_ingredients, num_ingredient_steps, ingredient_embedding_size = ingredient_embeddings.shape
            mask = torch.arange(torch.max(ingredient_lens)).unsqueeze(0) < ingredient_lens.unsqueeze(1)
            mask = mask.to(device)

            # --------------------- FORWARD PASS ---------------------
            outputs = torch.empty(batch_size, args.vocab_size, device=device)
            batch_loss = 0.0
            for t in range(min(num_ingredients, 13)):
                time_mask = torch.zeros(batch_size, num_ingredients, dtype=torch.bool)
                time_mask[:, :t+1] = True
                time_mask = time_mask.unsqueeze(2).unsqueeze(3).to(device)
                # get the ingredient embeddings until this timestep

                
                image_output, recipe_output = encoder(image_embeddings, title_embeddings, instruction_embeddings, ingredient_embeddings * time_mask)
                batch_loss += batch_triplet_loss(image_output, recipe_output.t())

                # use these to pass through the decoder and get the corresponding outputs
                outputs += decoder(torch.cat([image_output, recipe_output], dim=1))
                
                image_output = image_output.cpu().detach()
                recipe_output = recipe_output.cpu().detach()
                del image_output, recipe_output

            # Using this output, compute the average logits across ingredient sequences
            outputs = outputs / num_ingredients

            label_loss = F.binary_cross_entropy_with_logits(outputs, ingredient_indexes)
            batch_loss = batch_loss / num_ingredients

            loss = batch_loss + label_loss

            epoch_batch_triplet_loss += batch_loss.item()
            epoch_label_loss += label_loss.item()

        # --------------------- BACKWARD PASS ---------------------

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --------------------- WANDB LOGGING ----------------------

        if count % args.wandb_log_every == 0:
            wandb.log({
                'batch_loss': batch_loss.item(),
                'label_loss': label_loss.item(),
            })


        # --------------------- GARBAGE COLLECTION ---------------------
        
        del title_embeddings, ingredient_embeddings, ingredient_indexes, instruction_embeddings, image_embeddings, ingredient_lens, mask, loss, outputs, time_mask, batch_loss, label_loss
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device=device)
        torch.cuda.reset_max_memory_cached(device=device)


    return epoch_batch_triplet_loss, epoch_label_loss, count


def train():
    args = get_args()
    wandb.init(config=args, project="multimodal-sequential-autoencoder", entity="adhoki")

    # set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scaler = GradScaler()

    best_loss = float('inf')

    # load the data
    train_dataset = RecipeDataset(
        partition='val',
        ids_pkl='/home/ubuntu/recipe-dataset/val/val_keys.pkl', 
        cleaned_layers='/home/ubuntu/recipe-dataset/json/cleaned_layers.json',
        image_map='/home/ubuntu/recipe-dataset/json/image_map.json', 
        dataset_images='/home/ubuntu/recipe-dataset/val/', 
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

    val_dataset = RecipeDataset(
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        # change this later
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate(x, need_metadata=True)
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
    count = 0

    for i in range(args.epochs):
        epoch_batch_triplet_loss, epoch_label_loss, count = train_one_epoch(args, encoder, decoder, optimizer, train_loader, batch_triplet_loss, multi_label_loss, i, scaler, count)

        if args.evaluate_freq > 0 and i % args.evaluate_freq == 0:
            medr, recall, closest_image_dict = run_evaluation('im2recipe', encoder, train_loader, args.medr_N, device)
            wandb.log({
                'epoch': i,
                'medr': medr,
                'recall': recall,
            })
            print(f"EVALUATION: EPOCH: {i}, MEDR: {medr}, RECALL: {recall}")

        if args.save_freq > 0 and i % args.save_freq == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i,
                'batch_triplet_loss': epoch_batch_triplet_loss,
                'label_loss': epoch_label_loss,
                'args': args
            }, 'checkpoints/epoch_{}.pt'.format(i))

        if epoch_batch_triplet_loss + epoch_label_loss < best_loss:
            best_loss = epoch_batch_triplet_loss + epoch_label_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i,
                'batch_triplet_loss': epoch_batch_triplet_loss,
                'label_loss': epoch_label_loss,
                'args': args
            }, 'checkpoints/best/model.pt')

    return
    

train()