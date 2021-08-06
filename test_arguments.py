import argparse
import numpy as np
import os
import glob

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='WideResNet28_10', choices=['WideResNet28_10', 'resnet12', 'resnet12_rfs'])
    parser.add_argument('--training_method', type=str, default='S2M2_R',   help='rotation/S2M2_R')
    parser.add_argument('--save_dir', type=str, default='.', help='rotation/S2M2_R')
    parser.add_argument('--data_dir', type=str, default='', help = 'folder where datasets are stored (tieredImagenet)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--wrap_flag', type=int, default=0, metavar='N', help='make sure that you wrap the model only once')

    #parser.add_argument('--model_path', type=str, default='/home/michalislazarou/PhD/rfs_baseline/models/pretrained/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:0.5_b:0_trans_A_student1/resnet12_last.pth', help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'tieredimagenet', 'cifar', 'CUB'])
    #parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--which_dataset', type=str, default='images', choices=['images', 'pkl'])

    # specify data_root
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=100, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_unlabelled', type=int, default=30, metavar='N', help='Number of unlabelled in test')
    parser.add_argument('--n_aug_support_samples', default=0, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='Number of workers for dataloader')
    parser.add_argument('--unbalanced', type=bool, default=False, metavar='bool', help='Number of workers for dataloader')
    parser.add_argument('--distractor', type=bool, default=False, metavar='bool', help='Number of workers for dataloader')
    #parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')

    # algorithm parameter settings
    parser.add_argument('--reduce', type=str, default='none', choices=['isomap', 'itsa', 'mds', 'lle', 'se', 'pca', 'none'])
    parser.add_argument('--inference_semi', type=str, default='inductive_sk', choices=['transductive', 'inductive', 'inductive_sk'])
    parser.add_argument('--local', type=str, default='global', choices=['local', 'global'])
    parser.add_argument('--d', type=int, default=2, metavar='d', help='dimension of dimensionality reduction algorithm')
    parser.add_argument('--alpha', type=float, default=0.8, metavar='N', help='alpha used in graph diffusion')
    parser.add_argument('--K', type=int, default=20, metavar='N', help='Nearest neighbours to used in the Manifold creation')
    parser.add_argument('--T', type=float, default=3, metavar='N', help='power to raise probs matrix before sinkhorn algorithm')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='N', help='learning rate of fine-tuning')
    parser.add_argument('--inner_steps', type=int, default=5, metavar='N', help='fine-tuning steps')
    parser.add_argument('--denoising_iterations', type=int, default=1000, metavar='N', help='denoising steps')
    parser.add_argument('--beta_pt', type=float, default=0.5, metavar='N', help='power transform power')
    parser.add_argument('--ft_thresh', type=float, default=0, metavar='N', help='how many support examples in order to fine-tune')
    parser.add_argument('--best_samples', type=int, default=5, metavar='N', help='number of best samples per class chosen for pseudolabels')
    parser.add_argument('--semi_inference_method', type=str, default='inductive', choices=['transductive', 'inductive'])
    parser.add_argument('--finetune', type=str, default='none', choices=['finetuning_iter',  'finetuning_end', 'none'])
    parser.add_argument('--denoiser_type', type=str, default='linear', choices=['linear', 'last_layers'])
    parser.add_argument('--init_ft_iter', type=int, default=30, metavar='N', help='initial fine-tuning iterations')
    parser.add_argument('--sinkhorn_iter', type=int, default=1, metavar='N', help='sinkhorn iteration in optimal transport')
    parser.add_argument('--use_pt', type=str, default='pt_transform', choices=['pt_transform', 'no_pt_transform'])
    parser.add_argument('--use_ep', type=str, default='no_ep', choices=['ep', 'no_ep'])


    opt = parser.parse_args()

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data_pkl/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = False

    def get_assigned_file(checkpoint_dir, num):
        assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
        return assign_file

    def get_resume_file(checkpoint_dir):
        filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
        if len(filelist) == 0:
            return None

        filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
        return resume_file

    def get_best_file(checkpoint_dir):
        best_file = os.path.join(checkpoint_dir, 'best.tar')
        if os.path.isfile(best_file):
            return best_file
        else:
            return get_resume_file(checkpoint_dir)

    return opt
