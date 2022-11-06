# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data_closer.additional_transforms as add_transforms
from data_closer.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    def __init__(self, params, image_size,
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        if params.dataset == 'cifar':
            self.normalize_param = dict(mean = [0.5071, 0.4867, 0.4408], std = [0.2675, 0.2565, 0.2761])
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, args, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(args, image_size)
        self.dataset = args.dataset

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform, cub=self.dataset)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True, worker_init_fn= np.random.seed(34))
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, args, image_size):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = args.n_ways
        self.batch_size = args.n_shots + args.n_queries + args.n_unlabelled
        self.n_eposide = args.n_test_runs
        self.dataset = args.dataset
        self.model = args.model
        self.trans_loader = TransformLoader(args, image_size)


    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set

        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.dataset, self.model, data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        data_loader_params = dict(batch_sampler = sampler,  num_workers =12, pin_memory = True, worker_init_fn= np.random.seed(34) )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader



