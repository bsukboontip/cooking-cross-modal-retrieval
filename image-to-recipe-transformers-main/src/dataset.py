# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
from random import choice
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.utils import get_token_ids, list2Tensors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing

class Recipe1M(Dataset):
    """Dataset class for Recipe1M

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    transform : (callable, optional)
        A function/transform that takes in a PIL image and returns a transformed version.
    split : string
        Dataset split (train, val, or test).
    max_ingrs : int
        Maximum number of ingredients to use.
    max_instrs : int
        Maximum number of instructions to use.
    max_length_ingrs : int
        Maximum length of ingredient sentences.
    max_length_instrs : int
        Maximum length of instruction sentences.
    text_only_data : bool
        Whether to load paired or text-only samples.
    """

    def __init__(self, root, transform=None, split='train',
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 text_only_data=False,
                 load_actual_data=False):

        #load vocabulary
        self.vocab_inv = pickle.load(open('../data/vocab.pkl', 'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k
        # suffix to load text only samples or paired samples
        suf = '_noimages' if text_only_data else ''
        self.data = pickle.load(open('/home/ubuntu/recipe-dataset/traindata/test.pkl', 'rb'))
        print(f"total data: {len(self.data)}")

        # add functionality here to reduce the size of the dataset to include only indexes that are part of the analysis
        reduced_dataset_path = os.path.join(root, 'traindata', 'test.txt')
        if os.path.exists(reduced_dataset_path):
            with open(reduced_dataset_path, 'r') as file:
                self.reduced_list = file.read().splitlines()
                self.reduced_list = set(self.reduced_list)
        self.root = root
        
        self.ids = list(self.data.keys())
        
        print(len(self.ids))

        self.split = split
        self.transform = transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.text_only_data = text_only_data
        self.load_actual_data = load_actual_data

    def __getitem__(self, idx):

        entry = self.data[self.ids[idx]]

        if not self.text_only_data:
            # loading images
            if self.split == 'train':
                # if training, pick an image randomly
                img_name = choice(entry['images'])

            else:
                # if test or val we pick the first image
                img_name = entry['images'][0]

            img_name = '/'.join(img_name[:4])+'/'+img_name
            img = Image.open(os.path.join(self.root, self.split, img_name))
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = None

        title = entry['title']
        ingrs = entry['ingredients']
        instrs = entry['instructions']

        # turn text into indexes
        title_indexes = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_length_instrs])
        instrs_indexes = list2Tensors([get_token_ids(instr, self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]])
        ingrs_indexes = list2Tensors([get_token_ids(ingr, self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]])

        if self.load_actual_data:
            return img, title, title_indexes, ingrs, ingrs_indexes, instrs, instrs_indexes, self.ids[idx], img_name

        return img, title_indexes, ingrs_indexes, instrs_indexes, self.ids[idx], img_name

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data, load_actual_data):
    """ collate to consume and batchify recipe data
    """

    # Sort a data list by caption length (descending order).
    if load_actual_data:
        image, titles, title_indexes, ingrs, ingrs_indexes, instrs, instrs_indexes, ids, img_names = zip(*data)
    else:
        image, title_indexes, ingrs_indexes, instrs_indexes, ids, img_names = zip(*data)

    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)
    else:
        image = None
    title_targets = pad_input(title_indexes)
    ingredient_targets = pad_input(ingrs_indexes)
    instruction_targets = pad_input(instrs_indexes)

    if load_actual_data:
        return image, titles, title_targets, ingrs, ingredient_targets, instrs, instruction_targets, ids, img_names

    return image, title_targets, ingredient_targets, instruction_targets, ids, img_names


def get_loader(root, batch_size, resize, im_size, augment=False,
               split='train', mode='train',
               drop_last=True,
               text_only_data=False,
               load_actual_data=False):
    """Function to get dataset and dataloader for a data split

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    batch_size : int
        Batch size.
    resize : int
        Image size for resizing (keeps aspect ratio)
    im_size : int
        Image size for cropping.
    augment : bool
        Description of parameter `augment`.
    split : string
        Dataset split (train, val, or test)
    mode : string
        Loading mode (impacts augmentations & random sampling)
    drop_last : bool
        Whether to drop the last batch of data.
    text_only_data : type
        Whether to load text-only or paired samples.
    load_actual_data : bool
        Whether to load actual samples for title, ingredient, and instruction

    Returns
    -------
    loader : a pytorch DataLoader
    ds : a pytorch Dataset

    """

    transforms_list = [transforms.Resize((resize))]

    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(im_size))

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)

    ds = Recipe1M(root, transform=transforms_, split=split,
                  text_only_data=text_only_data, load_actual_data=load_actual_data)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        collate_fn=lambda b: collate_fn(b, load_actual_data), drop_last=drop_last)

    return loader, ds
