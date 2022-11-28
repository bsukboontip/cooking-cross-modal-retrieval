import os
import multiprocessing
from re import L
from dataset import get_loader
from models import get_model
import torch.backends.cudnn as cudnn
from config import get_args
from tqdm import tqdm
import torch
import numpy as np
import pickle
from utils.utils import load_checkpoint, count_parameters
import argparse
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'
random.seed(1234)


# load the dataset and dataloader...
# current implementation has the list of images to load already selected in the dataset file
def load_test_data():
    dataloader, dataset = get_loader(
        root='/home/ubuntu/recipe-dataset',
        batch_size=16,
        resize=224,
        im_size=224,
        augment=False, 
        split='test',
        mode='test',
        drop_last=False,
        load_actual_data=True
    )

    return dataset, dataloader


# manual hack to construct arguments for loading the model
def construct_model_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--tf_n_heads', type=int, default=4,
                        help='Number of attention heads in Transformer models.')
    parser.add_argument('--tf_n_layers', type=int, default=2,
                        help='Number of layers in Transformer models.')
    parser.add_argument('--hidden_recipe', type=int, default=512,
                        help='Embedding dimensionality for recipe representation.')
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Dimensionality of the output embeddings.')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone for the vision model')

    args = parser.parse_args()
    return args

def obtain_features(args, dataset, dataloader, checkpoints_dir, store_dict, device):

    vocab_size = len(dataset.get_vocab())
    model = get_model(args, vocab_size)
    print("recipe encoder", count_parameters(model.text_encoder))
    print("image encoder", count_parameters(model.image_encoder))

    _, model_dict, _ = load_checkpoint(checkpoints_dir, 'best', map_loc,
                                          store_dict)
    model.load_state_dict(model_dict, strict=True)
    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    image_features = recipe_features = None
    all_image_ids = []
    all_image_names = []
    all_titles = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image, titles, title_targets, ingrs, ingredient_targets, instrs, instruction_targets, ids, images = batch
        image = image.to(device)
        title_targets = title_targets.to(device)
        ingredient_targets = ingredient_targets.to(device)
        instruction_targets = instruction_targets.to(device)

        all_image_ids.extend(ids)
        all_image_names.extend(images)
        all_titles.extend(titles)

        with torch.no_grad():
            test_images = model.image_encoder(image)
            image_feat, recipe_feat, projected_feat = model(image, title_targets, ingredient_targets, instruction_targets)
        
        if image_features is None:
            image_features = image_feat.cpu().detach().numpy()
            recipe_features = recipe_feat.cpu().detach().numpy()
        else:
            image_features = np.vstack((image_features, image_feat.cpu().detach().numpy()))
            recipe_features = np.vstack((recipe_features, recipe_feat.cpu().detach().numpy()))

        if i == 62:
            break

    print(f"COMPLETED EXTRACTING FEATURES: FEATURE SIZE: {np.shape(image_features)}, {np.shape(recipe_features)}")

    return all_image_ids, image_features, recipe_features, all_image_names, all_titles


def visualize_results():
    pass


# take in as input the image and recipe features and corresponding image ids
# Do the following: 
# - Compute average medRank and recall rates for top-1, 5, 10
# - For each image - compute the top-10 matching image IDs
def run_evaluation(task_type, image_features, recipe_features, image_ids, titles, image_names, medr_N):
    indexes = np.argsort(image_ids)
    image_ids = [image_ids[i] for i in indexes]
    image_names = [image_names[i] for i in indexes]


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
        selected_images = [image_names[i] for i in ids]

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


def get_title(dataset, id):
    return dataset.data.get(id)['title']

def load_image(image_id):
    data_root_folder = '/home/ubuntu/recipe-dataset/test'
    image_path = os.path.join(data_root_folder, image_id)
    image = Image.open(image_path)
    return image

def visualize_results(id, image_dict, dataset, j):

    primary_image = image_dict['primary_image']
    selected_image_ids = image_dict['top_10_images'][:5]
    titles = image_dict['titles'][:5]
    
    fig = plt.figure(figsize=(25, 8))

    rows, cols = 1, 6

    fig.add_subplot(rows, cols, 1)
    image = load_image(primary_image)
    plt.imshow(image)
    plt.axis('off')
    plt.title('')


    for i, (image_id, title) in enumerate(zip(selected_image_ids, titles)):
        fig.add_subplot(rows, cols, i+2)
        image = load_image(image_id)
        plt.imshow(image)
        plt.axis('off')
        plt.title('')

    plt.savefig(f'test{j}.png')
    plt.close()

dataset, dataloader = load_test_data()

model_args = construct_model_args()
checkpoints_dir = '/home/ubuntu/cooking-cross-modal-retrieval/image-to-recipe-transformers-main/checkpoints/r50_ssl'
store_dict = {}

image_ids, image_features, recipe_features, image_names, titles = obtain_features(
    model_args, 
    dataset,
    dataloader,
    checkpoints_dir,
    store_dict,
    device
)

task_type = 'im2recipe'
average_median, global_recalls, closest_image_dict = run_evaluation(task_type, image_features, recipe_features, image_ids, titles, image_names, medr_N=1000)

# print(json.dumps(closest_image_dict, indent=4))

for i, (k, v) in enumerate(closest_image_dict.items()):
    visualize_results(id, v, dataset, i)