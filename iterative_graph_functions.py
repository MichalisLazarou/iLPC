from scipy.stats import t
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.weight_norm import WeightNorm
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ici_models.resnet12 import resnet12
import scipy as sp
import faiss

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores


def update_plabels(opt, support, support_ys, query):
    max_iter = 20
    no_classes = support_ys.max() + 1
    X = np.concatenate((support, query), axis=0).copy(order= 'C') # to bypass the error of array not C-contiguous
    k = opt.K
    # if opt.model == 'resnet12':
    #     k = X.shape[0]-1
    alpha = opt.alpha
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]

    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    D, I = index.search(X, k + 1)

    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - sp.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, no_classes))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(no_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0
    # --------try to filter-----------
    z_amax =-1*np.amax(Z, 1)[support_ys.shape[0]:]
    #-----------trying filtering--------
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1,1)

    p_labels[labeled_idx] = labels[labeled_idx]

    return p_labels[support.shape[0]:], probs_l1[support.shape[0]:], z_amax #p_probs #weights[support.shape[0]:]

def weight_imprinting(X, Y, model):
    no_classes = Y.max()+1
    imprinted = torch.zeros(no_classes, X.shape[1])
    for i in range(no_classes):
        idx = np.where(Y == i)
        tmp = torch.mean(X[idx], dim=0)
        tmp = tmp/tmp.norm(p=2)
        imprinted[i, :] = tmp
    model.weight.data = imprinted
    return model

def label_denoising(opt, support, support_ys, query, query_ys_pred):
    all_embeddings = np.concatenate((support, query), axis=0)
    input_size = all_embeddings.shape[1]
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    start_lr = 0.1
    end_lr = 0.1
    cycle = 50 #number of epochs
    step_size_lr = (start_lr - end_lr) / cycle
   # print(input_size, output_size.item())
    lambda1 = lambda x: start_lr - (x % cycle)*step_size_lr
    o2u = nn.Linear(input_size, output_size.item())
    o2u = weight_imprinting(torch.Tensor(all_embeddings[:support_ys.shape[0]]), support_ys, o2u)
    #o2u = weight_imprinting(torch.Tensor(all_embeddings[:opt.n_ways*opt.n_shots]), all_ys[:opt.n_ways*opt.n_shots], o2u)


    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    loss_statistics = torch.zeros(all_ys.shape, requires_grad=True)
    lr_progression =[]
    for epoch in range(opt.denoising_iterations):
        output = o2u(X)
        optimizer.zero_grad()
        loss_each = criterion(output, Y)
        loss_each = loss_each# * weights
        loss_all = torch.mean(loss_each)
        loss_all.backward()
        loss_statistics = loss_statistics + loss_each/(opt.denoising_iterations)
        optimizer.step()
        scheduler_lr.step()
        lr_progression.append(optimizer.param_groups[0]['lr'])
    return loss_statistics, lr_progression

def compute_optimal_transport(opt, M, epsilon=1e-6):
    #r is the P we discussed in paper r.shape = n_runs x total_queries, all entries = 1
    r = torch.ones(1, M.shape[0])
    #r = r * weights
    #c = torch.ones(1, M.shape[1]) * int(M.shape[0]/M.shape[1])
    c = torch.FloatTensor(opt.no_samples)
    idx = np.where(c.detach().cpu().numpy() <= 0)
    if opt.unbalanced == True:
        c = torch.FloatTensor(opt.no_samples)
        idx = np.where(c.detach().cpu().numpy() <=0)
        if len(idx[0])>0:
            M[:, idx[0]] = torch.zeros(M.shape[0], 1)

    M = M.cuda()
    r = r.cuda()
    c = c.cuda()
    M = torch.unsqueeze(M, dim=0)
    n_runs, n, m = M.shape
    P = M

    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    for i in range(opt.sinkhorn_iter):
        P = torch.pow(P, opt.T)
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if len(idx[0]) > 0:
                P[P != P] = 0
            if iters == maxiters:
                break
            iters = iters + 1
    P = torch.squeeze(P).detach().cpu().numpy()
    best_per_class = np.argmax(P, 0)
    if M.shape[1]==1:
        P = np.expand_dims(P, axis=0)
    labels = np.argmax(P, 1)
    return P, labels, best_per_class

def rank_per_class(no_cls, rank, ys_pred, no_keep):
    list_indices = []
    list_ys = []
    for i in range(no_cls):
        cur_idx = np.where(ys_pred == i)
        y = np.ones((no_cls,))*i
        class_rank = rank[cur_idx]
        class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
        class_rank_sorted[class_rank_sorted > no_keep] = 0
        indices = np.nonzero(class_rank_sorted)
        list_indices.append(cur_idx[0][indices[0]])
        list_ys.append(y)
    idxs = np.concatenate(list_indices, axis=0)
    ys = np.concatenate(list_ys, axis = 0)
    return idxs, ys


def iter_balanced(opt, support_features, support_ys, query_features, query_ys, labelled_samples, support_xs=None, query_xs =None, model=None):
    query_ys_updated = query_ys
    total_f = support_ys.shape[0] +query_ys.shape[0]
    #iterations = query_ys.shape[0]# int(query_ys.shape[0] / (opt.n_ways*opt.best_samples))
    iterations = int(query_ys.shape[0] / (opt.n_ways * opt.best_samples))
    for j in range(iterations):
        query_ys_pred, probs, weights = update_plabels(opt, support_features, support_ys, query_features)
        #------------------------probability pre-filtering for distractors-------------------------------------------------------------------
        rank = sp.stats.rankdata(weights, method='ordinal')
        #keep = (query_ys.shape[0] - 150)/opt.n_ways
        indices_pre, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, 30)
        #-----------------------------------------------------------------------------------------------------------

        P, query_ys_pred, indices = compute_optimal_transport(opt, torch.Tensor(probs))
        loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features[indices_pre], query_ys_pred[indices_pre])
        #loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features, query_ys_pred, weights=torch.tensor(weights))

        un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()#np.amax(P, 1) #
        un_loss_statistics = un_loss_statistics#*weights[support_ys.shape[0]:]
        rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')
        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred[indices_pre], opt.best_samples)
        #print(indices)
        indices = indices_pre[indices]
        #print(indices)
        #if len(indices)<5:
        #    break;
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
        if support_features.shape[0] == total_f or len(indices_pre) == 0:
            #print(support_features.shape[0], total_f)
            break
    #if opt.distractor == False:
    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    support_features = np.concatenate((support_features, query_features), axis=0)
    #query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    #query_ys_pred = support_ys[labelled_samples:]
    #query_ys = query_ys[query_ys_pred.shape[0]:]
    return support_ys, support_features


def iter_balanced_trans(opt, support_features, support_ys, query_features, query_ys, labelled_samples):
    query_ys_updated = query_ys
    total_f = support_ys.shape[0] +query_ys.shape[0]
    iterations =int(query_ys.shape[0])# int(opt.n_ways*opt.best_samples) #query_ys.shape[0]#
    #clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    #clf = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
    for j in range(iterations):
        # clf.fit(support_features, support_ys)
        # probs = clf.predict_proba(query_features)
        query_ys_pred, probs, weights = update_plabels(opt, support_features, support_ys, query_features)
        P, query_ys_pred, indices = compute_optimal_transport(opt, torch.Tensor(probs))

        loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features, query_ys_pred)
        un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()#np.amax(P, 1) #
        rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')

        #rank = sp.stats.rankdata(weights, method='ordinal')

        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, opt.best_samples)
        if len(indices)<5:
            break;
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
        #if opt.unbalanced:
        remaining_labels(opt, pseudo_ys)
        #print(sum(opt.no_samples), query_features.shape[0])
        if support_features.shape[0] == total_f:
            #print(support_features.shape[0], total_f)
            break
    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    support_features = np.concatenate((support_features, query_features), axis=0)
    query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    query_ys_pred = support_ys[labelled_samples:]
    query_ys = query_ys[query_ys_pred.shape[0]:]
    return query_ys, query_ys_pred

def remaining_labels(opt, selected_samples):
    #print(opt.no_samples)
    for i in range(len(opt.no_samples)):
        occurrences = np.count_nonzero(selected_samples == i)
        opt.no_samples[i] = opt.no_samples[i] - occurrences
        #opt.no_samples[opt.no_samples<0] = 0
    #print(opt.no_samples)


def im2features(X,Y, model):
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=False)
    tensor_list = []
    for batch_ndx, sample in enumerate(loader):
        x, _ = sample
        x = x.cuda()
        feat_support, _ = model(x)
        support_features = feat_support
        tensor_list.append(support_features.detach())
        torch.cuda.empty_cache()
    features = torch.cat(tensor_list, 0)
    return features


def pt_map_preprocess(support, query, beta):
    #X = torch.unsqueeze(torch.cat((torch.Tensor(support), torch.Tensor(query)), dim=0), dim=0)
    X = torch.unsqueeze(torch.cat((support, query), dim=0), dim=0)
    #X = scaleEachUnitaryDatas(X)
    #X = centerDatas(X)
    #nve_idx = np.where(X<0)
    #X[nve_idx] *=-1
    X = PT(X, beta)
    X = scaleEachUnitaryDatas(X)
    X = centerDatas(X)
    X = torch.squeeze(X)
    return X[:support.shape[0]], X[support.shape[0]:]

def check_chosen_labels(indices, ys_pred, y):
    correct = 0
    for i in range(indices.shape[0]):
        if ys_pred[i] == y[indices[i]]:
            correct = correct + 1
    return correct

#helper functions from PT-MAP

def centerDatas(datas):
    #PT code
#    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
 #   datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    # centre of mass of all data support + querries
    datas[:, :] -=datas[:, :].mean(1, keepdim=True)# datas[:, :, :] -
    norma = torch.norm(datas[:, :, :], 2, 2)[:, :, None].detach()
    datas[:, :, :] /= norma

    return datas


def scaleEachUnitaryDatas(datas):
    # print(datas.shape)
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas

def PT(datas, beta):
    datas[:, ] = torch.pow(datas[:, ] + 1e-6, beta)
    return datas

def preprocess_e2e(X, beta, params):
    X = torch.unsqueeze(X, dim=0)
    X = PT(X, beta)
    X = scaleEachUnitaryDatas(X)
    X = centerDatas(X)
    X = torch.squeeze(X)
    return X

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)

def load_pt_pretrained(params, model):
    model = model.to(params.device)
    cudnn.benchmark = True
    modelfile = get_resume_file(params.checkpoint_dir)
    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    if params.dataset == 'tieredImagenet':
        state.pop('module.classifier.L.weight_v', None)
        state.pop('module.classifier.L.weight_g', None)
    state_keys = list(state.keys())

    if params.wrap_flag == 0:
        callwrap = False
        #params.wrap_flag = 1
        if 'module' in state_keys[0]:
            callwrap = True
            params.wrap_flag = 1
        if callwrap:
            model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    return model

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def load_ici_pretrained(params, model):
    filename = "ici_models/"
    if params.dataset == 'CUB':
        filename = filename + "res12_cub.pth.tar"
    elif params.dataset == "tieredImagenet":
        filename = filename + "res12_tiered.pth.tar"
    elif params.dataset == 'miniImagenet':
        filename = filename + "res12_mini.pth.tar"
    else:
        filename = filename +"res12_cifar.pth.tar"
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.to(params.device)
    model.eval()
    return model

def dim_reduce(params, support, query):
    no_sup = support.shape[0]
    X = np.concatenate((support, query), axis = 0)
    X = normalize(X)
    if params.reduce == 'isomap':
        from sklearn.manifold import Isomap
        embed = Isomap(n_components=params.d)
    elif params.reduce == 'itsa':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=params.d, n_neighbors=5, method='ltsa')
    elif params.reduce == 'mds':
        from sklearn.manifold import MDS
        embed = MDS(n_components=params.d, metric=False)
    elif params.reduce == 'lle':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=params.d, n_neighbors=5, eigen_solver='dense')
    elif params.reduce == 'se':
        from sklearn.manifold import SpectralEmbedding
        embed = SpectralEmbedding(n_components=params.d)
    elif params.reduce == 'pca':
        from sklearn.decomposition import PCA
        embed = PCA(n_components=params.d)
    if params.reduce == 'none':
        X = X
    else:
        X = embed.fit_transform(X)
    return X[:no_sup].astype((np.float32)), X[no_sup:].astype((np.float32))
