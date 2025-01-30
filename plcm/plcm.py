import torch
import scipy
import numpy as np
from semi_distractive_wrn import artificial_balancing
import torch.nn.functional as F
from plcm.ssGMM import ss_GaussianMixture


steps=10

class Classifier(object):
    def __init__(self):
        self.initial_classifier()

    def fit(self, feature, label):
        self.classifier.fit(feature, label)

    def predict(self, feature, label=None):
        predicts = self.classifier.predict_proba(feature)
        if label is not None:
            pre_label = np.argmax(predicts, 1).reshape(-1)
            acc = np.mean(pre_label == label)
            return acc
        return predicts

    def initial_classifier(self):
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', class_weight='balanced', max_iter=1000)

def progressiveManager(args):
    ssGMM_ParameterGroup, min_lossGroup, max_lossGroup = [], [], []
    for i in range(steps):
    # for i in range(1):
        ss_GMM_parameterGroup, min_lossGroup, max_lossGroup = pseudoLossDistributionLearning(args, i, ssGMM_ParameterGroup, min_lossGroup, max_lossGroup)
        print('finished_iter:' , i)
    return ss_GMM_parameterGroup, min_lossGroup, max_lossGroup
    # pseudoLossConfidenceMetric(args, ss_GMM_parameterGroup, min_lossGroup, max_lossGroup)


def pseudoLossDistributionLearning(args, i, ssGMM_ParameterGroup, min_lossGroup, max_lossGroup):
    num_support = args.n_shots * args.n_ways
    num_query = args.n_queries * args.n_ways
    # num_unlabeled = args.n_unlabelled * args.n_ways
    import transductive_wrn as ts
    skc_lr = Classifier()
    print('process {}-step pseudo-loss distribution Learning'.format(i))
    query_select_0, query_select_1, query_unselect_0, query_unselect_1, unlabel_all, loss_assist_ALL = [], [], [], [], [], []
    n_shot = args.n_shots
    n_ways = args.n_ways
    if args.distractor:
        n_ways = args.n_ways + args.n_distractors
        print('classes of distractors: ', args.n_distractors)
    n_unlabelled = args.n_unlabelled
    if args.dataset =='CUB':
        n_unlabelled = 28
    n_queries = args.n_queries
    n_runs=1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_unlabelled#n_queries
    total_q = n_ways*n_queries
    n_samples = n_lsamples + n_usamples + total_q
    semi_train_samples = n_lsamples + n_usamples
    import FSLTask
    cfg = {'shot': n_shot, 'ways':n_ways, 'queries':n_queries+n_unlabelled}
    dataset = args.dataset+'_train'
    if args.model == 'resnet12':
        FSLTask.loadDataSet_res(dataset)
    else:
        FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    or_ndatas = FSLTask.GenerateRunSet(cfg=cfg).cuda()
    or_ndatas = or_ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1).cuda()
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs, n_shot+n_queries+n_unlabelled,n_ways).clone().view(n_runs, n_samples).cuda()
    print(or_ndatas.shape, labels.shape)
    n_runs=600
    ndatas = or_ndatas[:n_runs]
    labels = labels[:n_runs]
    # Power transform
    if args.model =='WideResNet28_10':
        beta = 0.5
        or_ndatas[:,] = torch.pow(or_ndatas[:,]+1e-6, beta)
    n_nfeat = or_ndatas.size(2)
    or_ndatas = ts.scaleEachUnitaryDatas(or_ndatas)
    if args.model =='WideResNet28_10':
        or_ndatas = ts.centerDatas(or_ndatas)
    X,  q = or_ndatas[:n_runs,:semi_train_samples,:], or_ndatas[:n_runs,semi_train_samples:,:]
    Y, q_ys = labels[:n_runs,:semi_train_samples], labels[:n_runs,semi_train_samples:]
    print(X.shape, q.shape, Y.shape, q_ys.shape)

    print("size of the datas...", X.size())

    # X = X.cpu()#.cuda()
    # Y = Y.cpu()#.cuda()
    print(X.shape)
    if args.distractor:
        # n_ways = params.n_ways
        from semi_distractive_wrn import distractor_features_pt
        X, Y, labelled_samples, q, q_ys = distractor_features_pt(args, X, Y, n_lsamples ,q, q_ys)
        print(X.shape, q.shape, Y.shape, q_ys.shape)
    # for i in range(30):
    for i in range(n_runs):
        if i % 100 == 0:
            print("plcm training: ", i)
        support_X, unlabel_X = X[i, :num_support].cpu(), X[i,num_support:].cpu()  # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_y, unlabel_y = Y[i, :num_support].cpu(), Y[i, num_support:].cpu()
        query_y, query_X = q_ys[i].cpu(), q[i].cpu()
        # print(support_X.shape, unlabel_X.shape, query_X.shape)
        assist_X = np.concatenate([unlabel_X, query_X])
        assist_y = np.concatenate([-1 * np.ones(unlabel_X.shape[0]), query_y])

        mix_X, mix_y = support_X, support_y
        select_query_index = np.array([])
        # num_select = int(args.n_unlabelled / steps)
        num_select = int(unlabel_y.shape[0]/(args.n_ways*steps))
        # print(unlabel_y.shape)
        # num_select = round(unlabel_y.shape[0]/(args.n_ways*steps))

        for i in range(len(ssGMM_ParameterGroup)):
            skc_lr.fit(mix_X, mix_y)

            pre_assist = skc_lr.predict(assist_X)
            assist_pseudoLabel = np.argmax(pre_assist, 1).reshape(-1)
            loss_assist = F.nll_loss(torch.log(torch.Tensor(pre_assist)), torch.Tensor(assist_pseudoLabel).long(), reduction='none')
            loss_assist = (loss_assist.numpy() - min_lossGroup[i]) / (max_lossGroup[i] - min_lossGroup[i])

            ssGMM_i = ss_GaussianMixture(ss_GMM_parameter=ssGMM_ParameterGroup[i])
            assist_InstancePredict = ssGMM_i.predict(loss_assist.reshape(-1, 1), proba=True)
            assist_InstanceLabel = np.argmax(assist_InstancePredict, 1).reshape(-1)
            assist_InstanceConfidence = np.max(assist_InstancePredict[:, 1::2], axis=1)

            select_assist_X, select_assist_y, select_index = [], [], []

            for class_item in range(args.n_ways):
                index_class_i = np.where(assist_pseudoLabel == class_item)
                select_index_classItem = index_class_i[0][
                    assist_InstanceConfidence[index_class_i].argsort()[::-1][:num_select * (i + 1)]]
                select_assist_X.extend(assist_X[select_index_classItem])
                select_assist_y.extend(assist_pseudoLabel[select_index_classItem])
                select_index.extend(select_index_classItem)

            select_assist_X = np.array(select_assist_X)
            select_assist_y = np.array(select_assist_y)
            select_index = np.array(select_index)

            select_query_index = select_index[select_index > unlabel_X.size(0)]

            mix_X = np.concatenate([support_X, select_assist_X])
            mix_y = np.concatenate([support_y, select_assist_y])

        unselect_query_index = np.setdiff1d(np.arange(unlabel_X.size(0), len(assist_y)), select_query_index)

        skc_lr.fit(mix_X, mix_y)

        pre_assist = skc_lr.predict(assist_X)
        assist_pseudoLabel = np.argmax(pre_assist, 1).reshape(-1)
        loss_assist = F.nll_loss(torch.log(torch.Tensor(pre_assist)), torch.Tensor(assist_pseudoLabel).long(),reduction='none')

        if len(ssGMM_ParameterGroup) > 0:
            query_select_0.extend(np.array(loss_assist)[select_query_index][assist_pseudoLabel[select_query_index] != assist_y[select_query_index]])
            query_select_1.extend(np.array(loss_assist)[select_query_index][assist_pseudoLabel[select_query_index] == assist_y[select_query_index]])

        query_unselect_0.extend(np.array(loss_assist)[unselect_query_index][assist_pseudoLabel[unselect_query_index] != assist_y[unselect_query_index]])
        query_unselect_1.extend(np.array(loss_assist)[unselect_query_index][assist_pseudoLabel[unselect_query_index] == assist_y[unselect_query_index]])
        unlabel_all.extend(np.array(loss_assist)[np.arange(0, unlabel_X.size(0))])
        loss_assist_ALL.extend(np.array(loss_assist))

    max_lossItem = max(loss_assist_ALL)
    min_lossItem = min(loss_assist_ALL)

    query_select_0 = (np.array(query_select_0) - min_lossItem) / (max_lossItem - min_lossItem)
    query_select_1 = (np.array(query_select_1) - min_lossItem) / (max_lossItem - min_lossItem)
    query_unselect_0 = (np.array(query_unselect_0) - min_lossItem) / (max_lossItem - min_lossItem)
    query_unselect_1 = (np.array(query_unselect_1) - min_lossItem) / (max_lossItem - min_lossItem)
    unlabel_all = (np.array(unlabel_all) - min_lossItem) / (max_lossItem - min_lossItem)

    x_labeled = np.concatenate([query_select_1, query_select_0, query_unselect_1, query_unselect_0])
    y_labeled = np.concatenate(
        [3 * np.ones(len(query_select_1)), 2 * np.ones(len(query_select_0)), np.ones(len(query_unselect_1)),
         np.zeros(len(query_unselect_0))])
    x_unlabeled = unlabel_all

    m_ssGaussianMixture = ss_GaussianMixture()
    ss_GMM_parameter = m_ssGaussianMixture.fit(x_labeled.reshape(-1, 1), y_labeled, x_unlabeled.reshape(-1, 1),
                                               beta=0.50, tol=0.1, max_iterations=20, early_stop='True')

    # unpade parameter
    ssGMM_ParameterGroup.append(ss_GMM_parameter)
    min_lossGroup.append(min_lossItem)
    max_lossGroup.append(max_lossItem)

    return ssGMM_ParameterGroup, min_lossGroup, max_lossGroup


def pseudoLossConfidenceMetric(args, ss_GMM_parameterGroup, min_lossGroup, max_lossGroup, X, Y, labelled_samples, q, q_ys):
    all_acc = []
    skc_lr = Classifier()
    # num_select = int(args.n_unlabelled / steps)
    for i in range(X.shape[0]):
        if i% 100==0:
            print("plcm: ", i)
        support_X, unlabel_X = X[i, :labelled_samples].cpu().detach().numpy(), X[i, labelled_samples:].cpu().detach().numpy() # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_y, unlabel_y = Y[i, :labelled_samples].cpu().detach().numpy(), Y[i, labelled_samples:].cpu().detach().numpy()
        query_y, query_X = q_ys[i].numpy(), q[i].numpy()
        num_select = int(unlabel_y.shape[0]/(args.n_ways*steps))
        # num_select = round(unlabel_y.shape[0] / (args.n_ways * steps))

    # print(num_select)
        mix_X, mix_y = support_X, support_y

        for i in range(len(ss_GMM_parameterGroup)):
            skc_lr.fit(mix_X, mix_y)

            pre_unlabel = skc_lr.predict(unlabel_X)
            unlabel_pseudoLabel = np.argmax(pre_unlabel, 1).reshape(-1)
            loss_unlabel = F.nll_loss(torch.log(torch.Tensor(pre_unlabel)), torch.Tensor(unlabel_pseudoLabel).long(),   reduction='none')
            loss_unlabel = (loss_unlabel.numpy() - min_lossGroup[i]) / (max_lossGroup[i] - min_lossGroup[i])

            ssGMM_i = ss_GaussianMixture(ss_GMM_parameter=ss_GMM_parameterGroup[i])
            unlabel_InstancePredict = ssGMM_i.predict(loss_unlabel.reshape(-1, 1), proba=True)
            unlabel_InstanceLabel = np.argmax(unlabel_InstancePredict, 1).reshape(-1)
            unlabel_InstanceConfidence = np.max(unlabel_InstancePredict[:, 1::2], axis=1)

            select_unalebl_X, select_unlabel_y, select_index = [], [], []

            for class_item in range(args.n_ways):
                index_class_i = np.where(unlabel_pseudoLabel == class_item)
                select_index_classItem = index_class_i[0][
                    unlabel_InstanceConfidence[index_class_i].argsort()[::-1][:num_select * (i + 1)]]
                select_unalebl_X.extend(unlabel_X[select_index_classItem])
                select_unlabel_y.extend(unlabel_pseudoLabel[select_index_classItem])
                select_index.extend(select_index_classItem)

            select_unalebl_X = np.array(select_unalebl_X)
            select_unlabel_y = np.array(select_unlabel_y)
            select_index = np.array(select_index)

            mix_X = np.concatenate([support_X, select_unalebl_X])
            mix_y = np.concatenate([support_y, select_unlabel_y])
        # print(mix_X.shape)
        if args.AN == 'yes':
            # print(mix_X.shape, mix_y.shape)
            mix_X, mix_y, _ = artificial_balancing(mix_X, mix_y, mix_y)
            # print(mix_X.shape, mix_y.shape)
        skc_lr.fit(mix_X, mix_y)

        query_acc = skc_lr.predict(query_X, query_y)
        all_acc.append(query_acc.tolist())
    return all_acc
    # m_true, h_true = mean_confidence_interval(all_acc)
    # print('Evaluation results: {:4f}, {:4f}'.format(m_true, h_true))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h