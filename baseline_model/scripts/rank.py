import os
import random
import numpy as np
import scripts.utils
import torchfile
import pickle
import sys
import json
sys.path.append("..")
# from ..args import get_parser
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

def print_helper(ingredients, recipes, test_img_name, img_file_name, true_id, true_cos_num, predicted_id=None):
    if test_img_name.get(img_file_name) != None:
        print(f"image_name: {img_file_name}")
        print(f"True recipe_id: {true_id}, Title: {recipes[true_id]['title']}, partition: {recipes[true_id]['partition']}")
        print(f"Ingredients: {recipes[true_id]['ingredients']}")
        print(f"Instructions: {recipes[true_id]['instructions']}")
        print(f"True cosine sim numerator: {true_cos_num}")
        print(f"-----------------------------------------------------")


random.seed(opts.seed)
type_embedding = opts.embtype 
test_img_name = dict()
# 'image'
# type_embedding = 'recipe'
print(opts.path_results)
with open(os.path.join(opts.path_results,'img_embeds.pkl'),'rb') as f:
    im_vecs = pickle.load(f)
with open(os.path.join(opts.path_results,'rec_embeds.pkl'),'rb') as f:
    instr_vecs = pickle.load(f)
with open(os.path.join(opts.path_results,'rec_ids.pkl'),'rb') as f:
    rec_names = pickle.load(f)
with open(os.path.join(opts.path_results,'img_ids.pkl'),'rb') as f:
    img_names = pickle.load(f)
with open(os.path.join(opts.path_results,'img_name.pkl'),'rb') as f:
    img_file_names = pickle.load(f)
with open(os.path.join(opts.data_path,'cleaned_ingredients.json'),'r') as f:
    ingredients = json.load(f)
with open(os.path.join(opts.data_path,'cleaned_layers.json'),'r') as f:
    recipes = json.load(f)
with open(os.path.join(opts.data_path,'test_img_names.txt'),'r') as f:
    for names in f:
        # Remove linebreak which is the last character of the string
        name = names[:-1]
        # Add item to the list
        test_img_name[name] = True

print(len(test_img_name))
print(type(ingredients), type(recipes))
print(ingredients['48ec3289d1'])
print(recipes['48ec3289d1'].keys())

# Sort based on names to always pick same samples for medr
idxs = np.argsort(rec_names)
rec_names = rec_names[idxs]

print(im_vecs.shape)
print(instr_vecs.shape)
print(rec_names.shape, rec_names[0])

# Ranker
N = opts.medr
idxs = range(N)

glob_rank = []
glob_recall = {1:0.0,5:0.0,10:0.0}
for i in range(10):

    ids = random.sample(range(0,len(rec_names)), N)
    im_sub = im_vecs[ids,:]
    instr_sub = instr_vecs[ids,:]
    ids_sub = rec_names[ids]
    img_ids_sub = img_file_names[ids]

    # if params.embedding == 'image':
    if type_embedding == 'image':
        sims = np.dot(im_sub,instr_sub.T) # for im2recipe
    else:
        sims = np.dot(instr_sub,im_sub.T) # for recipe2im
    
    # print(sims.shape, im_sub.shape, instr_sub.shape)

    med_rank = []
    recall = {1:0.0,5:0.0,10:0.0}

    for ii in idxs:

        name = ids_sub[ii]
        img_name = img_ids_sub[ii]
        # get a column of similarities
        sim = sims[ii,:]

        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()

        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)

        if (pos+1) == 1:
            recall[1]+=1
        if (pos+1) <=5:
            recall[5]+=1
        if (pos+1)<=10:
            recall[10]+=1
        if (pos+1) > 400 and (pos + 1) < 500:
            print_helper(ingredients, recipes, test_img_name, img_name, name, sim[ii])

        # store the position
        med_rank.append(pos+1)
    # print("N", N, recall)
    for i in recall.keys():
        recall[i]=recall[i]/N

    med = np.median(med_rank)

    for i in recall.keys():
        glob_recall[i]+=recall[i]
    glob_rank.append(med)

for i in glob_recall.keys():
    glob_recall[i] = glob_recall[i]/10
print("Mean median", np.average(glob_rank))
print("Recall", glob_recall)
