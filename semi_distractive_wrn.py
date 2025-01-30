import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import torch.optim as optim
from test_arguments import parse_option
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
import iterative_graph_functions as igf
from ici_models.ici import ICI
from sklearn import metrics
import scipy as sp
from scipy.stats import t


use_gpu = torch.cuda.is_available()

# ========================================
#      loading datas


def centerDatas(datas):
    mean = datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    return datas, mean

def scaleEachUnitaryDatas(datas):
   # print(datas.shape)
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


def distractor_features_pt(opt, ndatas, labels, labelled_samples, query_features, query_ys):
    support_features, unlabelled_features = ndatas[:, :labelled_samples], ndatas[:,labelled_samples:]  # X_pca[:labelled_samples], X_pca[labelled_samples:] #
    support_ys, unlabelled_ys = labels[:, :labelled_samples], labels[:, labelled_samples:]
    mask_support = ~support_ys.ge(opt.n_ways)
    support_ys = torch.masked_select(support_ys, mask_support)
    support_ys =support_ys.view(ndatas.size(0), -1)
    mask_features = mask_support.unsqueeze(2).expand(mask_support.size(0), mask_support.size(1), support_features.size(2))
    support_features = torch.masked_select(support_features, mask_features)
    support_features =support_features.view(ndatas.size(0), support_ys.size(1), -1)
    ndatas = torch.cat([support_features, unlabelled_features], dim=1)
    labels = torch.cat([support_ys, unlabelled_ys], dim=1)

    mask_query = ~query_ys.ge(opt.n_ways)
    query_ys = torch.masked_select(query_ys, mask_query)
    query_ys =query_ys.view(ndatas.size(0), -1)
    mask_features = mask_query.unsqueeze(2).expand(mask_query.size(0), mask_query.size(1), query_features.size(2))
    query_features = torch.masked_select(query_features, mask_features)
    query_features = query_features.view(ndatas.size(0), query_ys.size(1), -1)

    return ndatas, labels, opt.n_ways*opt.n_shots, query_features, query_ys

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
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self):
        if params.distractor:
            support = ndatas[:, :params.n_shots * params.n_ways, :]
            ys = labels[:, :params.n_shots * params.n_ways]
            labs = ys.reshape(n_runs, params.n_shots, params.n_ways)
            self.mus = support.reshape(n_runs, params.n_shots, params.n_ways, n_nfeat)[:, :n_shot, ].mean(1)
        else:
            self.mus = ndatas.reshape(n_runs, n_shot + n_unlabelled, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        #r is the P we discussed r.shape = n_runs x total_queries, all entries = 1
        r = r.cpu()#.cuda()
        # c is the q we discussed c.shape = n_runs x n_ways, all entries = 15
        c = c.cpu()#.cuda()
        #print(M[0,:5])
        n_runs, n, m = M.shape

        # doing the temperature T exponential here, M is distances
        P = torch.exp(- self.lam * M)
        #print(P[0, :5])
        #print("M and P shape: ", P.shape, M.shape, " n: ", n, " m: ", m)
        # adding up all the distances of  all the queries to every class center
        # and divide each component of P matrix with it, why??? is it to make it into probability
        #distribution
        # P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
        P = F.normalize(P, dim=2)
        #print(P.view((n_runs, -1)).shape)
        u = torch.zeros(n_runs, n).cpu()#.cuda()
        maxiters = 1000
        iters = 1

        # normalize this matrix
        # while torch.max(torch.abs(u - P.sum(2))) > epsilon:
        #    # print(r.shape, c.shape, u.shape, P.sum(1).shape, P.sum(2).shape)
        #     u = P.sum(2)
        #     #print(u[0])
        #     #print(r[0])
        #     #print(c[0])
        #     P *= (r / u).view((n_runs, -1, 1))
        #     P *= (c / P.sum(1)).view((n_runs, 1, -1))
        #     if iters == maxiters:
        #         break
        #     iters = iters + 1

        return P, torch.sum(P * M)
    
    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, params.n_ways) * n_usamples/params.n_ways
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
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
        return m, pm, olabels
    
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
        acc1, acc2, olabels = self.getAccuracy(op_xj)
        acc = [acc1, acc2]
        return acc, olabels

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def semi_ici(opt, X, Y, q, q_ys, labelled_samples):
    ici = ICI(classifier='lr', num_class=n_ways, step=3, reduce='pca', d=5)
    acc = []
    if params.distractor:
        X, Y, labelled_samples, q, q_ys = distractor_features_pt(params, X, Y, labelled_samples ,q, q_ys)
    for i in range(X.shape[0]):
        if i% 100==0:
            print("Iteration: ", i)
        support_features, unlabelled_features =  X[i,:labelled_samples], X[i,labelled_samples:] # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, unlabelled_ys = Y[i,:labelled_samples], Y[i,labelled_samples:]
        query_ys, query_features = q_ys[i], q[i]

        ici.fit(support_features, support_ys)
        unlabelled_features, unlabelled_ys_pred = ici.predict(unlabelled_features, None, False, unlabelled_ys, augmented_sup=True)

        support_features = np.concatenate((support_features, unlabelled_features), axis = 0)
        support_ys = np.concatenate((support_ys, unlabelled_ys_pred), axis = 0)
        if opt.truncate== 'yes':
            support_features, support_ys = truncate(support_features, support_ys)
        classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        classifier.fit(support_features, support_ys)
        query_ys_pred = classifier.predict(query_features)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)


def semi_plcm(opt, X, Y, q, q_ys, labelled_samples):
    acc = []
    if params.distractor:
        X, Y, labelled_samples, q, q_ys = distractor_features_pt(params, X, Y, labelled_samples ,q, q_ys)
    import plcm.plcm as plcm
    ss_GMM_parameterGroup, min_lossGroup, max_lossGroup = plcm.progressiveManager(opt)
    acc = plcm.pseudoLossConfidenceMetric(opt, ss_GMM_parameterGroup, min_lossGroup, max_lossGroup, X, Y, labelled_samples, q, q_ys)
    return mean_confidence_interval(acc)

def semi_ilpc(opt, X, Y, q, q_ys, labelled_samples):
    acc = []
    if params.distractor:
        X, Y, labelled_samples, q, q_ys = distractor_features_pt(params, X, Y, labelled_samples ,q, q_ys)
    for i in range(X.shape[0]):
        if i% 100==0:
            print("Iteration: ", i)
        support_features, unlabelled_features =  X[i,:labelled_samples].numpy(), X[i,labelled_samples:].numpy() # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, unlabelled_ys = Y[i,:labelled_samples].numpy(), Y[i,labelled_samples:].numpy()
        query_ys, query_features = q_ys[i].numpy(), q[i].numpy()
        opt.no_samples = np.array(np.repeat(float(unlabelled_ys.shape[0] / opt.n_ways), opt.n_ways))
        support_ys, support_features, w = igf.iter_balanced_server(opt, support_features, support_ys, unlabelled_features, unlabelled_ys, labelled_samples)
        if opt.distractor == True:
            keep = 1
        if opt.truncate== 'yes':
            support_features, support_ys, w = truncate(support_features, support_ys, w)
        classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)#, class_weight='balanced')
        if opt.ss_weights=='yes':
            classifier.fit(support_features, support_ys, sample_weight=w)
        else:
            classifier.fit(support_features, support_ys)
        query_ys_pred = classifier.predict(query_features)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)

def truncate(fs, ys, w):
    classes_list = np.unique(ys)
    Y = ys
    sup_ys = []
    features = []
    weights = []
    min = len(fs)
    for i in classes_list:
        idx = np.where(Y == i.item())
        fs_tmp = fs[idx]
        if min> len(fs_tmp):
            min =len(fs_tmp)
        features.append(fs_tmp)
        sup_ys.append(ys[idx])
        weights.append(w[idx])
    for i in range(len(sup_ys)):
        sup_ys[i] = sup_ys[i][:min]
        features[i] = features[i][:min]
        weights[i] = weights[i][:min]
    sup_ys, features, weights = np.concatenate(sup_ys),  np.concatenate(features), np.concatenate(weights)
    return features, sup_ys, weights


def get_prototypes(fs, ys):
    classes_list = torch.unique(ys)#.tolist()
    #no_classes = torch.max(ys).detach().cpu().numpy() + 1
    Y = ys.detach().cpu().numpy()
    prototypes = []
    for i in classes_list:
    #for i in range(no_classes):
        idx = np.where(Y == i.item())
        tmp = torch.unsqueeze(torch.mean(fs[idx], dim=0), 0)
        prototypes.append(tmp)
    prototypes = torch.cat(prototypes, dim = 0)
    return prototypes, classes_list

def distractor_features(opt, support_features, support_ys, unlabelled_features, unlabelled_ys, query_features, query_ys):
    idx_q = np.where(query_ys<params.n_ways)
    query_features, query_ys = query_features[idx_q], query_ys[idx_q]

    idx_sup = np.where(support_ys<params.n_ways)
    support_features, support_ys = support_features[idx_sup], support_ys[idx_sup]
    return support_features, support_ys, unlabelled_features, unlabelled_ys, query_features, query_ys

def semi_pt(opt, xs, ys, xq, yq, model=None):
    acc =[]
    for i in range(xs.shape[0]):
        if i %100 ==0:
            print("Iteration:", i)
        classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        support_features, query_features = xs[i,:], xq[i,:]
        support_ys, query_ys = ys[i,:], yq[i,:]
        classifier.fit(support_features, support_ys)
        query_ys_pred = classifier.predict(query_features)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)

if __name__ == '__main__':
    params = parse_option()
    np.random.seed(42)
# ---- data loading
    n_shot = params.n_shots
    n_ways = params.n_ways
    if params.distractor:
        n_ways = params.n_ways + params.n_distractors
        print('classes of distractors: ', params.n_distractors)
    n_unlabelled = params.n_unlabelled
    n_queries = params.n_queries
    params.model = 'WideResNet28_10'
    n_runs=1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_unlabelled#n_queries
    total_q = n_ways*n_queries
    n_samples = n_lsamples + n_usamples + total_q
    semi_train_samples = n_lsamples + n_usamples
    dataset = params.dataset

    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries+n_unlabelled}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    or_ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    or_ndatas = or_ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs, n_shot+n_queries+n_unlabelled,n_ways).clone().view(n_runs, n_samples)

    #partition for semi-supervised learning
    print(or_ndatas.shape, labels.shape)
    ndatas,  query_data = or_ndatas[:n_runs,:semi_train_samples,:], or_ndatas[:n_runs,semi_train_samples:,:]
    labels, q_labels = labels[:n_runs,:semi_train_samples], labels[:n_runs,semi_train_samples:]
    print(ndatas.shape, query_data.shape, labels.shape, q_labels.shape)

    # Power transform
    beta = 0.5
    ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)
    #print(ndatas.type())
    n_nfeat = ndatas.size(2)
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas, mean_support = centerDatas(ndatas)
    #trans-mean-sub

    query_data[:, ] = torch.pow(query_data[:, ] + 1e-6, beta)
    query_data = scaleEachUnitaryDatas(query_data)
    query_data = query_data-mean_support#centerDatas(query_data)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()
    print(ndatas.shape)
    
    if params.algorithm == 'ptmap':
        ndatas = ndatas.cpu()
        labels = labels.cpu()
        lam = 10
        if params.distractor:
            # n_ways = params.n_ways
            ndatas, labels, n_lsamples, query_data, q_labels = distractor_features_pt(params, ndatas, labels, n_lsamples, query_data, q_labels)

        model = GaussianModel(params.n_ways, lam)
        model.initFromLabelledDatas()
        alpha = 0.2
        optim = MAP(alpha)

        optim.verbose = False
        optim.progressBar = True
        acc_test, olabels = optim.loop(model, n_epochs=20)
        acc_pt, acc_std_pt = semi_pt(params, ndatas, olabels, query_data, q_labels, model=model)
        print('DATASET: {}, final accuracy semi_pt: {:0.2f} +- {:0.2f}, shots: {}, unlabelled: {}, AN: {}'.format(dataset, acc_pt * 100, acc_std_pt * 100, n_shot, n_unlabelled, params.AN))
    elif params.algorithm == 'ici':
        ndatas = ndatas.cpu()
        labels = labels.cpu()
        acc_ici, acc_std_ici = semi_ici(params, ndatas, labels, query_data, q_labels, n_lsamples)
        print('DATASET: {}, final accuracy ici: {:0.2f} +- {:0.2f}, shots: {}, unlabelled: {}, AN: {}'.format(dataset, acc_ici*100, acc_std_ici*100, n_shot, n_unlabelled, params.AN))
    elif params.algorithm == 'plcm':
        acc_plcm, acc_std_plcm = semi_plcm(params, ndatas, labels, query_data, q_labels, n_lsamples)
        print('DATASET: {}, final accuracy plcm: {:0.2f} +- {:0.2f}, shots: {}, unlabelled: {}, AN: {}'.format(dataset, acc_plcm * 100, acc_std_plcm * 100, n_shot, n_unlabelled, params.AN))
    elif params.algorithm == 'ilpc':
        ndatas = ndatas.cpu()
        labels = labels.cpu()
        acc_mine, acc_std = semi_ilpc(params, ndatas, labels, query_data, q_labels, n_lsamples)
        print('DATASET: {}, final accuracy ilpc: {:0.2f} +- {:0.2f}, shots: {}, unlabelled: {}, AN: {}, sinkhorn: {}, weights: {}, just_weights: {}, ss_weights : {}, n_distractors: {}'.format(dataset, acc_mine * 100, acc_std * 100, n_shot, n_unlabelled, params.AN, params.sinkhorn_bal, params.weights, params.just_weights, params.ss_weights, params.n_distractors))



    
    

