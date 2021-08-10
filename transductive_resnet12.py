from data_closer.datamgr import SetDataManager, TransformLoader
from test_arguments import parse_option
from models import res_mixup_model, wrn_mixup_model
import glob
import sys
#import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import numpy as np
import util
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
import iterative_graph_functions as igf
import scipy as sp
from scipy.stats import t
import os
from tqdm import tqdm
from sklearn import metrics
import eval.meta_eval as me
import configs
import random
import gc
import warnings
from ici_models.resnet12 import resnet12
from ici_models.ici import ICI
from rfs_models.util import create_model
import matplotlib.pyplot as plt
import eval.mixmatch_tools as mm_tools


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
        #in case the unlabelled are not 0
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
    elif params.model =='resnet12_rfs':
        if params.dataset == 'miniImagenet':
            num_classes = 64
            params.model_path = 'rfs_models/pretrained/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:0.5_b:0_trans_A_student1/resnet12_last.pth'
        elif params.dataset == 'tieredImagenet':
            num_classes = 351
            params.model_path = '/home/michalislazarou/PhD/rfs_baseline/models/pretrained/resnet12_tieredImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain100_epochs/ckpt_epoch_70.pth'
        model = create_model('resnet12', num_classes, params.dataset)
        ckpt = torch.load(params.model_path)
        model.load_state_dict(ckpt['model'])



    data_loader = dataloader_read(params, image_size)

    warnings.filterwarnings("ignore")

    test_acc, test_std = few_shot_test(params, model, data_loader)
    #sys.stdout = open("for_res_5shot.txt", "w")

    print('Dataset: {}, iterations: {}, pca_d:{}, best samples: {}, denoiser type: {}, Shots: {}, test_acc: {:.4f}, test_std: {:.4f}'.format(params.dataset, params.n_test_runs, params.d, params.best_samples, params.denoiser_type, params.n_shots, test_acc, test_std))
    print('Hyper Parameters: alpha: {}, k: {}, lr: {}, inner_steps:, {} T: {}, beta: {}, Fine-tuning type: {} best samples per class: {}, init_ft_steps: {}, finetune: {}, PT transform: {}'.format(params.alpha, params.K, params.lr, params.inner_steps, params.T, params.beta_pt, params.finetune, params.best_samples, params.init_ft_iter, params.finetune, params.use_pt))
    #sys.stdout.close()

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
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
    #list_hist = []
    #ici = ICI(classifier='lr', num_class=opt.n_ways, step=3, reduce=opt.reduce, d=opt.d)
    for idx, data in tqdm(enumerate(testloader)):
        query_ys, query_ys_pred = finetuning(opt, model, data)
        #query_ys, query_ys_pred = ici_test(opt, model, data, ici)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        #list_hist.append(my_list)
    #util.save_obj(list_hist, "uniform")
    #print(acc)
    return mean_confidence_interval(acc)

def finetuning(params, model, data):
    #1. convert to features
    # reset the model to the pre-trained network so that it is unbiased test
    with torch.no_grad():
        map = 'mi'
        if params.model == 'WideResNet28_10':
            model = igf.load_pt_pretrained(params, model)
        elif params.model == 'resnet12':
            model = igf.load_ici_pretrained(params, model)
        elif params.model == 'resnet12_rfs':
            map = 'rfs'
            ckpt = torch.load(params.model_path)
            model.load_state_dict(ckpt['model'])
        if params.which_dataset == 'images':
            data[1] = convert_to_few_shot_labels(data[1])
            n_way, _, height, width, channel = data[0].size()

            support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
            support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        elif params.which_dataset == 'pkl':
            support_xs, support_ys, query_xs, query_ys, = data
            n_way, _, height, width, channel = support_xs.size()

        support_xs = support_xs.contiguous().view(-1, height, width, channel)
        #plt.imshow(support_xs[0].cpu().permute(1, 2, 0))
        #plt.show()
        query_xs = query_xs.contiguous().view(-1, height, width, channel)

        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)
    #check for asupport augmentation
        if params.n_aug_support_samples > 1:
            if params.dataset == 'cifar':
                transform = TransformLoader(params, 32)
            else:
                transform = TransformLoader(params, 84)
            transform_function = transform.get_composed_transform(aug = True)
            support_xs, support_ys = igf.augment_examples(params, support_xs.detach().cpu().numpy(), support_ys.detach().cpu().numpy(), transform_function)

        support_features = me.im2features(support_xs, support_ys, model, map=map).cpu()
        query_features = me.im2features(query_xs, query_ys, model, map=map).cpu()

    #params pre-processing
        if params.use_pt =='pt_transform':
            X = igf.preprocess_e2e(torch.cat((support_features, query_features), dim=0), params.beta_pt, params)
            support_features, query_features = X[:support_features.shape[0]], X[support_features.shape[0]:]

    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()
    #print(support_features)
    #print(query_features)
    if params.use_ep == 'ep':
        support_features, query_features =me.update_embeddings(support_features, support_ys, query_features)
    if params.model == 'resnet12' or params.model =='resnet12_rfs':
        support_features, query_features = igf.dim_reduce(params, support_features, query_features)

    support_ys = support_ys.detach().cpu().numpy()#.astype(int)
    query_ys = query_ys.detach().cpu().numpy()

    labelled_samples = support_ys.shape[0]

    #2. iterative_graph_finetuing process
    if params.finetune == 'finetuning_end':
        #query_ys_pred, probs, weights = igf.update_plabels(params, support_features, support_ys, query_features)
        #_, query_ys_pred, _ = igf.compute_optimal_transport(params, torch.Tensor(probs))
        query_ys, query_ys_pred, query_xs = igf.iter_balanced(params, support_features, support_ys, query_features, query_ys, labelled_samples, support_xs, query_xs)
        #initial finetuning
        support_features, query_features = igf.init_ft(params, model, support_features, query_features, support_xs, support_ys, query_xs, query_ys_pred)#, weights)
        if params.reduce != 'none':
            support_features, query_features = igf.dim_reduce(params, support_features, query_features)
    classifier = nn.Linear(support_features.shape[1], support_ys.max() + 1)
    classifier = mm_tools.weight_imprinting(torch.tensor(support_features), torch.tensor(support_ys), classifier).cuda()
    query_ys_pred, probs, weights = igf.update_plabels(params, support_features, support_ys, query_features)
    support_features, query_features = igf.step_adapt(params, model, classifier, support_xs, support_ys, query_xs,query_ys_pred)
    query_ys, query_ys_pred, _= igf.iter_balanced(params, support_features, support_ys, query_features, query_ys, labelled_samples, support_xs, query_xs, model)
    # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    # clf.fit(support_features, support_ys)
    # query_ys_pred = clf.predict(query_features)
    #query_ys_pred, probs, weights = igf.update_plabels(params, support_features, support_ys, query_features)
    #_, query_ys_pred, _ = igf.compute_optimal_transport(params, torch.Tensor(probs))
    #query_ys, query_ys_pred, _= igf.iter_balanced_local(params, support_features, support_ys, query_features, query_ys, labelled_samples, support_xs, query_xs, model)

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

        if params.n_aug_support_samples > 1:
            if params.dataset == 'cifar':
                transform = TransformLoader(params, 32)
            else:
                transform = TransformLoader(params, 84)
            transform_function = transform.get_composed_transform(aug = True)
            support_xs, support_ys = igf.augment_examples(params, support_xs.detach().cpu().numpy(), support_ys.detach().cpu().numpy(), transform_function)

        support_features = me.im2features(support_xs, support_ys, model, map='mi').cpu()
        query_features = me.im2features(query_xs, query_ys, model, map='mi').cpu()

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
        from rfs_dataset.transform_cfg import transforms_test_options, transforms_list
        if params.dataset == 'miniImagenet':
            from rfs_dataset.mini_imagenet import MetaImageNet
            train_trans, test_trans = transforms_test_options[params.transform]
            data_loader = DataLoader(MetaImageNet(args=params, partition='test',train_transform=train_trans,test_transform=test_trans,fix_seed=False),
                                         batch_size=1, shuffle=False, drop_last=False, num_workers=params.num_workers)
        elif params.dataset == 'tieredImagenet':
            from rfs_dataset.tiered_imagenet import MetaTieredImageNet
            train_trans, test_trans = transforms_test_options[params.transform]
            data_loader = DataLoader(MetaTieredImageNet(args=params, partition='test', train_transform=train_trans, test_transform=test_trans,fix_seed=False),
                                         batch_size=1, shuffle=False, drop_last=False,num_workers=params.num_workers)
        elif params.dataset == 'cifar':
            from rfs_dataset.cifar import MetaCIFAR100
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
