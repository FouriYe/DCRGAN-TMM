import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init


from termcolor import cprint
from time import gmtime, strftime
import numpy as np
from sklearn import preprocessing
import argparse
import os
import glob
import random
from sklearn.metrics.pairwise import cosine_similarity
from model import _param, _SRN, _netMetric
from triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss
import util
import dataload

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--dataroot', default='../data/',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume',  type=str, help='the model to resume')

parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval',  type=int, default=40)
parser.add_argument('--SRN',  type=str)
parser.add_argument('--MN',  type=str)
parser.add_argument('--epoch',  type=int, default=180000)
parser.add_argument('--method', default='', type=str)
parser.add_argument('--rec_attSize', type=int, default=10)
parser.add_argument('--feat', default='vis', type=str, help='vis/ att+vis')
parser.add_argument('--useWD',  action='store_true', default=False, help='use Weight Decay?')

opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA  = 5
opt.REG_W_LAMBDA = 0.001
opt.Adv_LAMBDA = 1
opt.useSR = False

opt.lr = 0.0001
if opt.dataset == "APY":
    opt.batch_size = 64
else:
    opt.batch_size = 64  # 512

if opt.dataset == "CUB" or opt.dataset == "FLO":
    opt.z_dim = 100
else:
    opt.z_dim = 10
lmd1 = 1.#for triplet loss of MN
lmd2 = 0.00005#for weight decay of MN
lmd3 = 1.#for MSE loss of SRN
lmd4 = 0.000005#for weight decay of SRN
if not opt.useWD:
    lmd2 = 0#for weight decay of MN

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

def train():
    data = dataload.DATA_LOADER(opt)
    opt.attSize = data.text_dim
    opt.resSize = data.feature_dim
    
    
    print("train class: ")
    print(data.train_cls_num)
    print("test class: ")
    print(data.test_cls_num)
    SRN = _SRN(opt).cuda()
    SRN.apply(weights_init)
    MN = _netMetric(opt).cuda()
    MN.apply(weights_init)

    exp_info = 'Metric_{}'.format(opt.dataset)
    if opt.preprocessing:
        exp_params = 'SRN_Preprocessing_'+opt.method+str(opt.rec_attSize)+"_"+opt.feat
    else:
        exp_params = 'SRN_NonPreprocessing_'+opt.method+str(opt.rec_attSize)+"_"+opt.feat
    if opt.useWD:
        exp_params = exp_params+'_useWD'
    else:
        exp_params = exp_params+'_noWD'
    out_dir  = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    #cprint(" The output dictionary is {}".format(out_subdir), 'red')
    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir_MN = out_subdir + '/log_MN_{:s}_{}.txt'.format(exp_info, opt.exp_idx)
    with open(log_dir_MN, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    
    optimizerSRN = optim.Adam(SRN.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerMN = optim.Adam(MN.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    """Step1: train MN """
    if opt.MN == None:
        for it in range(0, 100000+1):
            if opt.feat == "vis":
                batch_feature, batch_label, _ = data.next_batch(opt.batch_size)
                v_feat = batch_feature.cuda()             # image data
                y_true = util.map_label(batch_label, data.seenclasses).cuda()
                aug_s_feat = MN(v_feat)
                #loss
                if opt.method == "hard":
                    triplet_loss = batch_hard_triplet_loss(feature = aug_s_feat,labels = y_true,margin = 1.)
                elif opt.method == "all":
                    triplet_loss,_ = batch_all_triplet_loss(feature = aug_s_feat,labels = y_true,margin = 1.)
                reg_loss1 = Variable(torch.Tensor([0.0])).cuda()
                for name, p in MN.named_parameters():
                            if 'weight' in name:
                                reg_loss1 += p.pow(2).sum()
                
                all_loss = lmd1*triplet_loss + lmd2*reg_loss1
                all_loss.backward()
                optimizerMN.step()
                MN.zero_grad()
                if it%100 == 0 and it>0:
                    log_text = "it {}: Triplet_loss: {:.4f} Reg_loss1: {:.4f}(weighted: {:.4f})\n".format(it,triplet_loss.data,reg_loss1.data[0],reg_loss1.data[0]*lmd2)
                    print(log_text)
                    with open(log_dir_MN, 'a') as f:
                        f.write(log_text+'\n')
            elif opt.feat =="att+vis":
                batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
                v_feat = batch_feature.cuda()             # image data
                y_true = util.map_label(batch_label, data.seenclasses).cuda()
                text_feat = batch_att.cuda()
                feats = torch.cat((v_feat,text_feat), dim=1)
                aug_s_feat = MN(feats)
                #loss
                if opt.method == "hard":
                    triplet_loss = batch_hard_triplet_loss(feature = aug_s_feat,labels = y_true,margin = 1.)
                elif opt.method == "all":
                    triplet_loss,_ = batch_all_triplet_loss(feature = aug_s_feat,labels = y_true,margin = 1.)
                reg_loss1 = Variable(torch.Tensor([0.0])).cuda()
                for name, p in MN.named_parameters():
                    if 'weight' in name:
                        reg_loss1 += p.pow(2).sum()
                
                all_loss = lmd1*triplet_loss + lmd2*reg_loss1
                all_loss.backward()
                optimizerMN.step()
                MN.zero_grad()
                if it%100 == 0 and it>0:
                    log_text = "it {}: Triplet_loss: {:.4f} Reg_loss1: {:.4f}(weighted: {:.4f})\n".format(it,triplet_loss.data,reg_loss1.data[0],reg_loss1.data[0]*lmd2)
                    print(log_text)
                    with open(log_dir_MN, 'a') as f:
                        f.write(log_text+'\n')
        save_model_MN(it, MN, out_subdir +'/model_MN_Iter_{:d}.tar'.format(it))
    else:
        if os.path.isfile(opt.MN):
            print("=> loading checkpoint '{}'".format(opt.MN))
            checkpoint = torch.load(opt.MN)
            MN.load_state_dict(checkpoint['state_dict_MetricNet'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.MN))
    MN.eval()
    """Step2: get class-level augmented semantics """
    scaler = preprocessing.MinMaxScaler()
    if opt.feat == "vis":
        labels = util.map_label(data.train_label, data.seenclasses).cuda()
        v_feat = data.train_feature.cuda()
        feats = v_feat
        all_augmented_s_feat = MN(feats).cpu().detach().numpy()
    elif opt.feat =="att+vis":
        labels = util.map_label(data.train_label, data.seenclasses).cuda()
        v_feat = data.train_feature.cuda()
        train_att_np = data.train_att.numpy()
        text_feat = np.array([ train_att_np[i,:] for i in labels])
        text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
        feats = torch.cat((v_feat,text_feat),dim=1)
        all_augmented_s_feat = MN(feats).cpu().detach().numpy()
    _train_feature = torch.from_numpy(scaler.fit_transform(all_augmented_s_feat)).float().cuda()
    mx = _train_feature.max()
    _train_feature = _train_feature.mul_(1/mx)
    data.augmented_s_feat = np.zeros([data.seenclasses.shape[0], opt.rec_attSize], np.float32)
    for i in range(data.seenclasses.shape[0]):
        v_feat = data.train_feature.cuda()
        labels = util.map_label(data.train_label, data.seenclasses).cuda()
        augmented_s_feat = _train_feature[labels == i]
        data.augmented_s_feat[i] = np.mean(augmented_s_feat.cpu().detach().numpy(), axis=0)
    
    """Step3: train SRN """
    augmented_s_feat = Variable(torch.from_numpy(data.augmented_s_feat.astype('float32'))).cuda()
    if opt.SRN == None:
        for it in range(0, 100000+1):
            #data，
            batch_feature, batch_label, _ = data.next_batch(opt.batch_size)
            v_feat = batch_feature.cuda()             # image data
            y_true = util.map_label(batch_label, data.seenclasses).cuda()
            text_feat = data.train_att.cuda()

            rectified_s_feat_ = SRN(text_feat)
            #loss
            MSE_loss = Variable(torch.Tensor([0.0])).cuda()
            MSE_loss += (rectified_s_feat_ - augmented_s_feat).pow(2).sum().sqrt()
            MSE_loss *= 1.0/data.train_cls_num
            
            reg_loss2 = Variable(torch.Tensor([0.0])).cuda()
            for name, p in SRN.named_parameters():
                if 'weight' in name:
                    reg_loss2 += p.pow(2).sum()
            all_loss = lmd3*MSE_loss + lmd4*reg_loss2
            all_loss.backward()
            optimizerSRN.step()
            SRN.zero_grad()
            if it%5000==0:
                save_model_SRN(it, SRN, out_subdir +'/model_SRN_Iter_{:d}.tar'.format(it))
                log_text = "it {}: MSE_loss: {:.4f} Reg_loss2: {:.4f}(weighted: {:.4f})\n".format(it,MSE_loss.data[0],reg_loss2.data[0],lmd4*reg_loss2.data[0])
                print(log_text)
                with open(log_dir_MN, 'a') as f:
                    f.write(log_text+'\n')
    else:
        if os.path.isfile(opt.SRN):
            print("=> loading checkpoint '{}'".format(opt.SRN))
            checkpoint = torch.load(opt.SRN)
            SRN.load_state_dict(checkpoint['state_dict_RecNet'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.SRN))

def save_model_MN(it, MN, fout):
    torch.save({
        'it': it + 1,
        'state_dict_MetricNet': MN.state_dict()
    }, fout)

def save_model_SRN(it, SRN, fout):
    torch.save({
        'it': it + 1,
        'state_dict_RecNet': SRN.state_dict()
    }, fout)

def matrix_weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
    
def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)
        
if __name__ == "__main__":
    train()