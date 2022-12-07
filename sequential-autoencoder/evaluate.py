import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import random
from dataset import RecipeDataset, collate
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.config import get_args
from models.encoder import CombinedEncoder


def run_forward(encoder, batch, device):
    
    with autocast(dtype=torch.float16):
        title_embeddings = batch['title_embeddings'].type(torch.float16).to(device)
        ingredient_embeddings = batch['ingredient_embeddings'].type(torch.float16).to(device)
        ingredient_indexes = batch['ingredient_indexes'].type(torch.float16).to(device)
        instruction_embeddings = batch['instruction_embeddings'].type(torch.float16).to(device)
        # here it is still true

        image_embeddings = batch['image_embeddings'].type(torch.float16).to(device)

        batch_size, num_ingredients, num_ingredient_steps, ingredient_embedding_size = ingredient_embeddings.shape

        # We just need the image and recipe embeddings here

        image_output, recipe_output = encoder(image_embeddings, title_embeddings, instruction_embeddings, ingredient_embeddings)

        return image_output, recipe_output

"""
Run evaluation on the validation set and test set, and return the metrics
Additionally, return the loss and the closest image dictionary

Note: valid dataloader contains collate function with need_metadata=True
"""
def run_evaluation(task_type, encoder, valid_dataloader, medr_N, device):

    # image_features, recipe_features, image_ids, titles, image_names

    # load the model
    encoder.eval()
    image_ids = []
    titles = []

    # obtain the image features frmo the validation set, until medr_N samples are obtained
    batch = next(iter(valid_dataloader))
    image_features, recipe_features = run_forward(encoder, batch, device)
    image_ids.extend(batch['image_ids'])
    print("IMAGE IDS\n", image_ids)
    titles.extend(batch['titles'])
    image_features, recipe_features = image_features.cpu().detach().numpy(), recipe_features.cpu().detach().numpy()
    
    while len(image_ids) < medr_N * 2:
        print(f"NOW: {len(image_ids)}, medR: {medr_N * 2}")
        batch = next(iter(valid_dataloader))
        image_features_batch, recipe_features_batch = run_forward(encoder, batch, device)
        image_features = np.concatenate((image_features, image_features_batch.detach().cpu().numpy()), 0)
        recipe_features = np.concatenate((recipe_features, recipe_features_batch.detach().cpu().numpy()), 0)
        image_ids.extend(batch['image_ids'])
        titles.extend(batch['titles'])

    # sort the image ids
    indexes = np.argsort(image_ids)
    image_ids = [image_ids[i] for i in indexes]
    # image_names = [image_names[i] for i in indexes]


    n = medr_N
    idxs = range(n)

    global_ranks = []
    global_recall = {1: 0.0, 5:0.0, 10:0.0}
    closest_image_dict = {}

    # repeat the exercise for 10 times
    for i in range(10):
        ids = random.sample(range(0, len(image_ids)), n)
        image_subfeatures = image_features[ids, :]
        recipe_subfeatures = recipe_features[ids, :]
        selected_ids = [image_ids[i] for i in ids]
        selected_images = [image_ids[i] for i in ids]

        if task_type == 'im2recipe':
            similarities = np.dot(image_subfeatures, recipe_subfeatures.T)
        elif task_type =='recipe2im':
            similarities = np.dot(recipe_subfeatures, image_subfeatures.T)
        else:
            raise ValueError('Invalid task type')

        median_ranks = []
        recalls = {1: 0.0, 5:0.0, 10:0.0}

        for idx in idxs:
            name = selected_ids[idx]

            similarity = similarities[idx, :]
            sorting = np.argsort(similarity)[::-1].tolist()
            pos = sorting.index(idx)

            closest_image_dict[name] = {
                'primary_image': selected_images[idx],
                'top_10_images': [selected_images[i] for i in sorting[:10]],
                'titles': [titles[i] for i in sorting[:10]],
                'similarity_scores': [str(i) for i in similarity[sorting[:10]]]
            }

            if pos == 0:
                recalls[1] += 1
            if pos <= 5:
                recalls[5] += 1
            if pos <= 10:
                recalls[10] += 1
            
            median_ranks.append(pos)

        for k in recalls:
            recalls[k] = recalls[k] / n 

        median = np.median(median_ranks)
        for k in recalls:
            global_recall[k] += recalls[k]
        
        global_ranks.append(median)

    for k in global_recall:
        global_recall[k] = global_recall[k] / 10

    print(f"MEAN MEDIAN: {np.average(global_ranks)}")
    print(f"RECALL: {global_recall}")

    return np.average(global_ranks), global_recall, closest_image_dict


if __name__ == '__main__':

    args = get_args()

    valid_dataset = RecipeDataset(
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

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        # change this later
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate(x, need_metadata=True)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_type = 'im2recipe'
    encoder = CombinedEncoder()
    encoder.load_state_dict(torch.load('checkpoints/best/encoder.pt'))
    encoder = encoder.to(device)
    medr_N = 100
    run_evaluation(task_type, encoder, valid_dataloader, medr_N, device)

