# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import pickle
import os
identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity, cub = 'mini'):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.crop = False
        if cub == 'CUB':
            self.crop = True
            self.bbox = load_obj('/home/michalislazarou/PhD/filelists/CUB/bbox')
            self.images_number = load_obj('/home/michalislazarou/PhD/filelists/CUB/images_number')


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        if self.crop:
            img = self.crop_bb(img, image_path)
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def crop_bb(self, img, image_path):
        image_key = image_path.replace('/home/michalislazarou/PhD/filelists/CUB/CUB_200_2011/images/', '')
        bbox_key = self.images_number[image_key]
        bbox_coords = tuple(self.bbox[bbox_key])#[int(x*1.15) for x in self.bbox[bbox_key]]
        left = bbox_coords[0]
        right = bbox_coords[2]+bbox_coords[0]
        upper = bbox_coords[3]+bbox_coords[1]
        lower = bbox_coords[1]
        improved = tuple([left, lower, right, upper])
        #print(bbox_coords, improved)
        #cropped1 = img.crop(bbox_coords)
        #cropped1.show()

        cropped2 = img.crop(improved)
        #cropped2.show()

        return cropped2

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self,dataset, model, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False, worker_init_fn= np.random.seed(34))
        for cl in self.cl_list:
            sub_dataset = SubDataset(dataset, model,self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, dataset, model, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.crop = False
        if dataset == 'CUB' and model == 'resnet12':
            self.crop = True
            self.bbox = load_obj('/home/michalislazarou/PhD/filelists/CUB/bbox')
            self.images_number = load_obj('/home/michalislazarou/PhD/filelists/CUB/images_number')


    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        if self.crop:
            img = self.crop_bb(img, image_path)
        img = self.transform(img)
        #print(img.shape)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

    def crop_bb(self, img, image_path):
        image_key = image_path.replace('/home/michalislazarou/PhD/filelists/CUB/CUB_200_2011/images/', '')
        bbox_key = self.images_number[image_key]
        bbox_coords = tuple(self.bbox[bbox_key])#[int(x*1.15) for x in self.bbox[bbox_key]]
        left = bbox_coords[0]
        right = bbox_coords[2]+bbox_coords[0]
        upper = bbox_coords[3]+bbox_coords[1]
        lower = bbox_coords[1]
        improved = tuple([left, lower, right, upper])
        #print(bbox_coords, improved)
        #cropped1 = img.crop(bbox_coords)
        #cropped1.show()

        cropped2 = img.crop(improved)
        #cropped2.show()

        return cropped2

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            #print(i, self.n_episodes)
            #print(torch.randperm(self.n_classes)[0])
            yield torch.randperm(self.n_classes)[:self.n_way]


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)