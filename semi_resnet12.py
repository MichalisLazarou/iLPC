from data_closer.datamgr import SetDataManager
from test_arguments import parse_option
from models import res_mixup_model, wrn_mixup_model
import glob
import torch.backends.cudnn as cudnn
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
from ici_models.ici import ICI
from sklearn import metrics
#import eval.meta_eval as me
import configs
import random
import gc
from ici_models.resnet12 import resnet12
#import eval.mixmatch_tools as mm_tools


def main():
    seed = 42
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
    if params.dataset == 'cifar':
        image_size = 32
    else:
        image_size = 84

    # print(loadfile)

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

    data_loader = dataloader_read_ss(params, image_size)

    test_acc, test_std = few_shot_test(params, model, data_loader)
    print('Dataset: {}, iterations: {}, best samples: {}, algorithm: {}, Shots: {}, unlabelled: {}, test_acc: {:.4f}, test_std: {:.4f}'.format(params.dataset, params.n_test_runs, params.best_samples, params.algorithm, params.n_shots, params.n_unlabelled, test_acc, test_std))



def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def few_shot_test(opt, model, testloader):
    model = model.eval()
    model.cuda()
    acc = []
    #ici = ICI(classifier='lr', num_class=opt.n_ways, step=3, reduce=opt.reduce, d=opt.d)
    for idx, data in tqdm(enumerate(testloader)):
        if opt.algorithm == 'ilpc':
            query_ys, query_ys_pred = ss_finetuning(opt, model, data)
        elif opt.algorithm =='ici':
            ici = ICI(classifier='lr', num_class=opt.n_ways, step=3, reduce='pca', d=opt.d)
            query_ys, query_ys_pred = ici_test_ss(opt, model, data, ici)
        else:
            print("This algorithm is not available in this experiment")
            break;
        #query_ys, query_ys_pred = ss_finetuning(opt, model, data)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)


def ss_finetuning(params, model, data):
    # 1. convert to features
    # reset the model to the pre-trained network so that it is unbiased test
    if params.model == 'WideResNet28_10':
        model = igf.load_pt_pretrained(params, model)
    elif params.model == 'resnet12':
        model = igf.load_ici_pretrained(params, model)
    if params.which_dataset == 'images':
        data[1] = convert_to_few_shot_labels(data[1])
        n_way, _, height, width, channel = data[0].size()

        support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
        support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        if params.n_unlabelled > 0:
            unlabelled_ys, query_ys = query_ys[:, :params.n_unlabelled], query_ys[:, params.n_unlabelled:]
            unlabelled_xs, query_xs = query_xs[:, :params.n_unlabelled], query_xs[:, params.n_unlabelled:]
    elif params.which_dataset == 'pkl':
        support_xs, support_ys, query_xs, query_ys, unlabelled_xs, unlabelled_ys, _ = data
        n_way, _, height, width, channel = support_xs.size()

    support_xs = support_xs.contiguous().view(-1, height, width, channel)
    query_xs = query_xs.contiguous().view(-1, height, width, channel)
    unlabelled_xs = unlabelled_xs.contiguous().view(-1, height, width, channel)

    support_ys = support_ys.contiguous().view(-1)
    query_ys = query_ys.contiguous().view(-1)
    unlabelled_ys = unlabelled_ys.contiguous().view(-1)

    support_features = igf.im2features(support_xs, support_ys, model).cpu()
    query_features = igf.im2features(query_xs, query_ys, model).cpu()
    unlabelled_features = igf.im2features(unlabelled_xs, unlabelled_ys, model).cpu()
    params.no_samples = np.array(np.repeat(float(unlabelled_ys.shape[0] / params.n_ways), params.n_ways))
    # params pre-processing
    if params.model == 'WideResNet28_10':#pt_transform':
        query_features = torch.cat((query_features, unlabelled_features), dim = 0)
        support_features, query_features = igf.pt_map_preprocess(support_features, query_features, params.beta_pt)
        query_features, unlabelled_features = query_features[:params.n_queries*params.n_ways], query_features[params.n_queries*params.n_ways:]
    #print(support_features.shape, query_features.shape, unlabelled_features.shape)

    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()
    unlabelled_features.detach().cpu().numpy()

    support_ys = support_ys.detach().cpu().numpy()
    query_ys = query_ys.detach().cpu().numpy()
    unlabelled_ys = unlabelled_ys.detach().cpu().numpy()

    if params.model == 'resnet12':
        query_features = np.concatenate((query_features, unlabelled_features), axis = 0)
        support_features, query_features = igf.dim_reduce(params, support_features, query_features)
        query_features, unlabelled_features = query_features[:params.n_queries*params.n_ways], query_features[params.n_queries*params.n_ways:]

    labelled_samples = support_ys.shape[0]
    support_ys, support_features = igf.iter_balanced(params, support_features, support_ys, unlabelled_features,unlabelled_ys, labelled_samples)
    #    unlabelled_ys, unlabelled_ys_pred = igf.iter_balanced_trans(params, support_features, support_ys, unlabelled_features, unlabelled_ys, labelled_samples)
    #unlabelled_features = igf.im2features(unlabelled_xs, torch.Tensor(unlabelled_ys), model).detach().cpu().numpy()
    #unlabelled_ys_pred, probs, weights = igf.update_plabels(params, support_features, support_ys, unlabelled_features)
    #_, unlabeleld_ys_pred, _ = igf.compute_optimal_transport(params, torch.Tensor(probs))
    #support_features = np.concatenate((support_features, unlabelled_features), axis=0)
    #support_ys = np.concatenate((support_ys, unlabelled_ys_pred), axis=0)
    #labelled_samples = support_ys.shape[0]
    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    clf.fit(support_features, support_ys)
    query_ys_pred = clf.predict(query_features)
    # 3. without fine-tuning
    # query_ys, query_ys_pred = igf.iter_balanced(params, support_features, support_ys, query_features, query_ys, labelled_samples)

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


def ici_test_ss(params, model, data, ici):
    #1. convert to features
    # reset the model to the pre-trained network so that it is unbiased test
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
            if params.n_unlabelled > 0:
                unlabelled_ys, query_ys = query_ys[:, :params.n_unlabelled], query_ys[:, params.n_unlabelled:]
                unlabelled_xs, query_xs = query_xs[:, :params.n_unlabelled], query_xs[:, params.n_unlabelled:]
        elif params.which_dataset == 'pkl':
            support_xs, support_ys, query_xs, query_ys, unlabelled_xs, unlabelled_ys, _ = data
            n_way, _, height, width, channel = support_xs.size()

        support_xs = support_xs.contiguous().view(-1, height, width, channel)
        query_xs = query_xs.contiguous().view(-1, height, width, channel)
        unlabelled_xs = unlabelled_xs.contiguous().view(-1, height, width, channel)

        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)
        unlabelled_ys = unlabelled_ys.contiguous().view(-1)

        support_features = igf.im2features(support_xs, support_ys, model).cpu()
        query_features = igf.im2features(query_xs, query_ys, model).cpu()
        unlabelled_features = igf.im2features(unlabelled_xs, unlabelled_ys, model).cpu()

        if params.model == 'WideResNet28_10':  # pt_transform':
            query_features = torch.cat((query_features, unlabelled_features), dim=0)
            support_features, query_features = igf.pt_map_preprocess(support_features, query_features, params.beta_pt)
            query_features, unlabelled_features = query_features[:params.n_queries * params.n_ways], query_features[params.n_queries * params.n_ways:]


    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()
    unlabelled_features.detach().cpu().numpy()

    support_ys = support_ys.detach().cpu().numpy()
    query_ys = query_ys.detach().cpu().numpy()
    #unlabelled_ys = unlabelled_ys.detach().cpu().numpy()

    #labelled_samples = support_ys.shape[0]
    ici.fit(support_features, support_ys)
    query_ys_pred = ici.predict(query_features, unlabelled_features, False, query_ys)

    return query_ys, query_ys_pred


def dataloader_read_ss(params, image_size):
    if params.which_dataset == 'images':
        datamgr = SetDataManager(params, image_size)
        split = 'novel'
        loadfile = configs.data_dir[params.dataset] + split + '.json'
        data_loader = datamgr.get_data_loader(loadfile, aug=False)
    else:
        from pkl_dataset.transform_cfg import transforms_test_options, transforms_list
        if params.dataset == 'miniImagenet':
            from pkl_dataset.mini_imagenet import SS2MetaImageNet
            train_trans, test_trans = transforms_test_options[params.transform]
            data_loader = DataLoader(SS2MetaImageNet(args=params, partition='test', train_transform=train_trans, test_transform=test_trans, fix_seed=False),
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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def convert_to_few_shot_labels(data):
    for i in range(data.shape[0]):
        data[i, :] = i
    return data


if __name__ == '__main__':
    main()
