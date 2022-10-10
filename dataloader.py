import torch.utils.data as data
from PIL import Image
import os
import pickle
import numpy as np
import lmdb
import torch
import sys
import json

from torchvision import transforms

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(..., file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')
       
class ImageLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None, clean_ingredients='', clean_layers=''):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        with open(clean_ingredients, 'r') as f:
            self.ingredients = json.load(f)
        
        with open(clean_layers, 'r') as f:
            self.recipes = json.load(f)

    def __getitem__(self, index):
        recipId = self.ids[index]
        # we force 80 percent of them to be a mismatch
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        target = match and 1 or -1

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1'))
        sample = pickle.loads(serialized_sample,encoding='latin1')
        imgs = sample['imgs']

        # image
        if target == 1:
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
            else:
                imgIdx = 0

            loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            # path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])
            path = os.path.join(self.imgPath, loader_path, imgs[imgIdx]['id'])
        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[rndindex].encode('latin1'))

            rndsample = pickle.loads(serialized_sample,encoding='latin1')
            rndimgs = rndsample['imgs']

            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
            else:
                imgIdx = 0

            loader_path = [rndimgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.imgPath, loader_path, rndimgs[imgIdx]['id'])
            # path = self.imgPath + rndimgs[imgIdx]['id']

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # load image
        img = self.loader(path)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'] - 1
        rec_id = self.ids[index]

        if target == -1:
            img_class = rndsample['classes'] - 1
            img_id = self.ids[rndindex]
        else:
            img_class = sample['classes'] - 1
            img_id = self.ids[index]

        # read from JSON files and load text ingredients + instructions
        ingredients = self.ingredients[recipId]
        title = self.recipes[recipId]['title']
        instructions = self.recipes[recipId]['instructions']

        # output
        if self.partition == 'train':
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln, title, ingredients, instructions], [target, img_class, rec_class]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln, title, ingredients, instructions], [target]
        else:
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln, title, ingredients, instructions], [target, img_class, rec_class, img_id, rec_id]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln, title, ingredients, instructions], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)

def print_batch(input, target):
    print("INPUT:", len(input), "TARGET:", len(target))
    print(f"IMAGE: {input[0].shape}\n\nINSTRUCTIONS: {input[1].shape}\n\nINSTRUCTIONS LENGTH: {input[2]}\n\nINGREDIENTS: {input[3]}\n\nINGREDIENTS LENGTH: {input[4]}\n\nTITLE: {input[5]}\n\nINGREDIENTS: {input[6]}\n\nINSTRUCTIONS: {input[7]}\n\n")
    print(f"TARGET: {target[0]}\n\nIMAGE ID: {target[1]}\n\nRECIPE ID: {target[2]}\n\n")
    print("\n\n-----------------------------------------------------------------------------------------------------------------------------------\n\n")

# data_loader = data.DataLoader(
#     ImageLoader(
#         '../val/',
#         transform=transforms.Compose([
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]),
#         data_path='../val',
#         partition='val'),
#     batch_size=2,
#     shuffle=False
# )