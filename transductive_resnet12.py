from data_closer.datamgr import SetDataManager, TransformLoader
from test_arguments import parse_option
from models import res_mixup_model, wrn_mixup_model
import glob
import sys
#import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
import iterative_graph_functions as igf
import scipy as sp
from scipy.stats import t
import os
from tqdm import tqdm
from sklearn import metrics
import configs
import random
import gc
import warnings
from ici_models.resnet12 import resnet12
from ici_models.ici import ICI
#import eval.mixmatch_tools as mm_tools


def main():
    seed = 33
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    params = parse_option()
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
    params.save_dir, params.dataset, params.model, params.training_method)

    if params.n_unlabelled>0:
        params.n_unlabelled = 0
    if params.dataset == 'cifar':
        image_size = 32
    else:
        image_size = 84

    if params.model == 'WideResNet28_10':
        if params.dataset == 'cifar':
            model = wrn_mixup_model.wrn28_10(64, loss_type='softmax')
        else:
            model = wrn_mixup_model.wrn28_10()
        model = model.to(params.device)
        model = igf.load_pt_pretrained(params, model)
    elif params.model == 'resnet12':
        if params.dataset == 'CUB':
            num_classes = 100
        elif params.dataset == 'tieredImagenet':
            num_classes = 351
        else:
            image_size=84
            num_classes = 64
        model = resnet12(num_classes)

    data_loader = dataloader_read(params, image_size)

    warnings.filterwarnings("ignore")
    test_acc, test_std = few_shot_test(params, model, data_loader)

    print('Dataset: {}, iterations: {}, best samples: {}, algorithm: {}, Shots: {}, test_acc: {:.4f}, test_std: {:.4f}'.format(params.dataset, params.n_test_runs, params.best_samples, params.algorithm, params.n_shots, test_acc, test_std))


def few_shot_test(opt, model, testloader):
    model = model.eval()
    model.cuda()
    acc = []
    for idx, data in tqdm(enumerate(testloader)):
        if opt.algorithm == 'ilpc':
            query_ys, query_ys_pred = finetuning(opt, model, data)
        elif opt.algorithm =='ici':
            ici = ICI(classifier='lr', num_class=opt.n_ways, step=3, reduce='pca', d=opt.d)
            query_ys, query_ys_pred = ici_test(opt, model, data, ici)
        else:
            print("This algorithm is not available in this experiment")
            break;
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)

def finetuning(params, model, data):
    with torch.no_grad():
        if params.model == 'WideResNet28_10':
            model = igf.load_pt_pretrained(params, model)
        elif params.model == 'resnet12':
            model = igf.load_ici_pretrained(params, model)
        if params.which_dataset == 'images':
            data[1] = convert_to_few_shot_labels(data[1])
            n_way, _, height, width, channel = data[0].size()

            support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
            support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        elif params.which_dataset == 'pkl':
            support_xs, support_ys, query_xs, query_ys, = data
            n_way, _, height, width, channel = support_xs.size()

        support_xs = support_xs.contiguous().view(-1, height, width, channel)
        query_xs = query_xs.contiguous().view(-1, height, width, channel)

        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)

        support_features = igf.im2features(support_xs, support_ys, model).cpu()
        query_features = igf.im2features(query_xs, query_ys, model).cpu()
        params.no_samples = np.array(np.repeat(float(query_ys.shape[0] / params.n_ways), params.n_ways))

        if params.model == 'WideResNet28_10':
            #if params.use_pt =='pt_transform':
            X = igf.preprocess_e2e(torch.cat((support_features, query_features), dim=0), params.beta_pt, params)
            support_features, query_features = X[:support_features.shape[0]], X[support_features.shape[0]:]

    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()

    if params.model == 'resnet12':
        support_features, query_features = igf.dim_reduce(params, support_features, query_features)

    support_ys = support_ys.detach().cpu().numpy()
    query_ys = query_ys.detach().cpu().numpy()

    labelled_samples = support_ys.shape[0]

    query_ys, query_ys_pred = igf.iter_balanced_trans(params, support_features, support_ys, query_features, query_ys, labelled_samples)
    return query_ys, query_ys_pred

def convert_features(X, Y, model):
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=False)
    tensor_list = []
    for batch_ndx, sample in enumerate(loader):
        x, _ = sample
        feat_support, _ = model(x)
        support_features = feat_support.view(x.size(0), -1)
        tensor_list.append(support_features.detach())
        gc.collect()
        torch.cuda.empty_cache()
    features = torch.cat(tensor_list, 0)
    return features

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def ici_test(params, model, data, ici):
    #1. convert to features
    # reset the model to the pre-trained network so that it is unbiased test
    with torch.no_grad():
        if params.model == 'WideResNet28_10':
            model = igf.load_pt_pretrained(params, model)
        elif params.model == 'resnet12':
            model = igf.load_ici_pretrained(params, model)
        if params.which_dataset =='images':
            data[1] = convert_to_few_shot_labels(data[1])
            n_way, _, height, width, channel = data[0].size()

            support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
            support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        elif params.which_dataset == 'pkl':
            support_xs, support_ys, query_xs, query_ys, = data
            n_way, _, height, width, channel = support_xs.size()

        support_xs = support_xs.contiguous().view(-1, height, width, channel)
        query_xs = query_xs.contiguous().view(-1, height, width, channel)

        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)

        support_features = igf.im2features(support_xs, support_ys, model).cpu()
        query_features = igf.im2features(query_xs, query_ys, model).cpu()

    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()

    support_ys = support_ys.detach().cpu().numpy().astype(int)
    query_ys = query_ys.detach().cpu().numpy()

    labelled_samples = support_ys.shape[0]
    ici.fit(support_features, support_ys)
    query_ys_pred = ici.predict(query_features, None, False, query_ys)

    return query_ys, query_ys_pred


def dataloader_read(params, image_size):
    if params.which_dataset == 'images':
        datamgr = SetDataManager(params, image_size)
        split = 'novel'
        loadfile = configs.data_dir[params.dataset] + split + '.json'
        data_loader = datamgr.get_data_loader(loadfile, aug=False)
    else:
        from pkl_dataset.transform_cfg import transforms_test_options, transforms_list
        if params.dataset == 'miniImagenet':
            from pkl_dataset.mini_imagenet import MetaImageNet
            train_trans, test_trans = transforms_test_options[params.transform]
            data_loader = DataLoader(MetaImageNet(args=params, partition='test',train_transform=train_trans,test_transform=test_trans,fix_seed=False),
                                         batch_size=1, shuffle=False, drop_last=False, num_workers=params.num_workers)
        elif params.dataset == 'tieredImagenet':
            from pkl_dataset.tiered_imagenet import MetaTieredImageNet
            train_trans, test_trans = transforms_test_options[params.transform]
            data_loader = DataLoader(MetaTieredImageNet(args=params, partition='test', train_transform=train_trans, test_transform=test_trans,fix_seed=False),
                                         batch_size=1, shuffle=False, drop_last=False,num_workers=params.num_workers)
        elif params.dataset == 'cifar':
            from pkl_dataset.cifar import MetaCIFAR100
            train_trans, test_trans = transforms_test_options['D']
            data_loader = DataLoader(MetaCIFAR100(args=params, partition='test', train_transform=train_trans, test_transform=test_trans, fix_seed=False),
                                         batch_size=1, shuffle=False, drop_last=False, num_workers=params.num_workers)
    return data_loader

def convert_to_few_shot_labels(data):
    for i in range(data.shape[0]):
        data[i,:] = i
    return data


if __name__ == '__main__':
    main()
    #sys.stdout = open("for_res_5shot.txt", "w")
