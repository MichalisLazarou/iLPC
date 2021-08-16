import collections
import pickle
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
from test_arguments import parse_option
import iterative_graph_functions as igf
from ici_models.ici import ICI
from sklearn import metrics
import scipy as sp
from scipy.stats import t

#import my_code as michalis


use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas


def centerDatas(datas):
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]

    # centre of mass of all data support + querries
    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    return datas

def scaleEachUnitaryDatas(datas):
   # print(datas.shape)
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cpu()
        
    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)                           

    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        #r is the P we discussed r.shape = n_runs x total_queries, all entries = 1
        r = r.cpu()
        # c is the q we discussed c.shape = n_runs x n_ways, all entries = 15
        c = c.cpu()
        #print(M[0,:5])
        n_runs, n, m = M.shape

        # doing the temperature T exponential here, M is distances
        P = torch.exp(- self.lam * M)
        #P = torch.exp(-1*M)
        #print(P[0, :5])
        #print("M and P shape: ", P.shape, M.shape, " n: ", n, " m: ", m)
        # adding up all the distances of  all the queries to every class center
        # and divide each component of P matrix with it, why??? is it to make it into probability
        #distribution
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
        #print(P.view((n_runs, -1)).shape)
        u = torch.zeros(n_runs, n).cpu()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
           # print(r.shape, c.shape, u.shape, P.sum(1).shape, P.sum(2).shape)
            u = P.sum(2)
            #print(u[0])
            #print(r[0])
            #print(c[0])
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1

        #u = P.sum(2)
        #P *= (r / u).view((n_runs, -1, 1))
        return P, torch.sum(P * M)
    
    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        #print(p_xj_test.shape)
        p_xj[:, n_lsamples:] = p_xj_test
        
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
     
        p_xj = model.getProbas()
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, n_epochs=20):
        
        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def trans_ici(opt, X, Y, labelled_samples):
    ici = ICI(classifier='lr', num_class=n_ways, step=3, reduce='pca', d=5)
    acc = []
    for i in range(X.shape[0]):
        if i% 400==0:
            print("ici: ", i)
        support_features, query_features =  X[i,:labelled_samples], X[i,labelled_samples:] # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, query_ys = Y[i,:labelled_samples], Y[i,labelled_samples:]
        if params.unbalanced == True:
            query_features, query_ys, opt.no_samples = unbalancing(opt, query_features, query_ys)
        ici.fit(support_features, support_ys)
        query_ys_pred = ici.predict(query_features, None, False, query_ys)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)

def trans_ilpc(opt, X, Y, labelled_samples):
    acc = []
    for i in range(X.shape[0]):
        if i% 400==0:
            print("ilpc: ", i)
        support_features, query_features =  X[i,:labelled_samples], X[i,labelled_samples:] # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, query_ys = Y[i,:labelled_samples], Y[i,labelled_samples:]
        labelled_samples = support_features.shape[0]
        if params.unbalanced == True:
            query_features, query_ys, opt.no_samples = unbalancing(opt, query_features, query_ys)
        else:
            opt.no_samples = np.array(np.repeat(float(query_ys.shape[0]/opt.n_ways),opt.n_ways))

        #query_ys_pred, probs, _ = igf.update_plabels(opt, support_features, support_ys, query_features)
        #P, query_ys_pred, indices = igf.compute_optimal_transport(opt, torch.Tensor(probs))
        query_ys, query_ys_pred = igf.iter_balanced_trans(opt, support_features, support_ys, query_features, query_ys, labelled_samples)
        #clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        #clf = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        #clf.fit(support_features, support_ys)
        #query_ys_pred = clf.predict(query_features)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)

def unbalancing(opt, query_features, query_ys):
    max = opt.n_queries
    min = opt.n_queries - opt.un_range
    no_samples = np.array(np.random.randint(min, max, size=opt.n_ways))
    no_classes = query_ys.max() + 1
    q_y =[]
    q_f = []
    for i in range(no_classes):
        idx = np.where(query_ys == i)
        tmp_y, tmp_x = query_ys[idx], query_features[idx]
        #print(tmp_y[0:no_samples[i]].shape)
        q_y.append(tmp_y[0:no_samples[i]])
        q_f.append(tmp_x[0:no_samples[i]])
    q_y = torch.cat(q_y, dim=0)
    q_f = torch.cat(q_f, dim=0)
    #print(q_y.shape)
    return q_f, q_y, no_samples
    # print(model.weight.shape, imprinted.shape)


if __name__ == '__main__':
# ---- data loading
    params = parse_option()
    n_shot = params.n_shots
    n_ways = params.n_ways
    n_unlabelled = params.n_unlabelled
    n_queries = params.n_queries
    print(params.n_queries, params.unbalanced)
    if params.unbalanced == True:
        params.un_range = 10
        params.n_queries = n_queries + params.un_range
        n_queries = params.n_queries
    print(params.n_queries, params.unbalanced)
    n_runs=1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    dataset = params.dataset

    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    print(ndatas.shape)
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    print(params.unbalanced)


    # Power transform
    beta = 0.5
    #------------------------------------PT-MAP-----------------------------------------------
    nve_idx = np.where(ndatas.cpu().detach().numpy()<0)
    ndatas[nve_idx] *= -1
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas[nve_idx]*=-1 # return the sign
    #------------------------------------------------------------------------------------------
    print(ndatas.type())
    n_nfeat = ndatas.size(2)
    #

    ndatas_icp = ndatas
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas)
    #ndatas = ndatas.cpu()
    #labels = labels.cpu()

    print("size of the datas...", ndatas.size())

    if params.algorithm =='ptmap':
        lam = 10
        model = GaussianModel(n_ways, lam)
        model.initFromLabelledDatas()

        alpha = 0.2
        optim = MAP(alpha)

        optim.verbose=False
        optim.progressBar=True

        acc_test = optim.loop(model, n_epochs=20)
        print("final accuracy PT-MAP: {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))
    elif params.algorithm == 'ilpc':
        acc_mine, acc_std = trans_ilpc(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy ilpc: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_mine * 100,acc_std * 100, n_shot,n_queries))
    elif params.algorithm =='ici':
        ndatas_icp = ndatas_icp.cpu()
        acc_ici, acc_std_ici = trans_ici(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy ici: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_ici * 100, acc_std_ici * 100, n_shot, n_queries))
    else:
        print('Algorithm not supported!')
    

