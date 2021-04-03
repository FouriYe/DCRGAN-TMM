import argparse
import os
import random
import sys
import time
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import classifier
import classifier2    #for searched repre
import model
import soft_cls
import util
import torch
import numpy as np
import glob
import dataload

def loadPretrainedMain(netS, savePost):
    print('Loading pretrained Mainnet......')
    path = './pretrain/'
    netS.load_state_dict( torch.load( path+savePost, map_location='cuda:0' ) )

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/home/yezihan/code/TMM-DCRGAN/data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--encoderSize', type=int, default=312, help='size of encoded size')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')

parser.add_argument('--loss_syn_num', type=int, default=30, help='G learning rate')
parser.add_argument('--cyc_seen_weight', type=float, default=1, help='weight of the seen class cycle loss')
parser.add_argument('--cyc_unseen_weight', type=float, default=1, help='weight of the unseen class cycle loss')
parser.add_argument('--cyc_seen_weight2', type=float, default=1e-4, help='weight of the seen class cycle loss')
parser.add_argument('--cyc_unseen_weight2', type=float, default=1e-4, help='weight of the unseen class cycle loss')

parser.add_argument('--cls_syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--cls_batch_size', type=int, default=5, help='G learning rate')
parser.add_argument('--f_hid', type=int, default=4096, help='forward hidden units')

parser.add_argument('--new_lr', type=int, default=0, help='forward hidden units')
parser.add_argument('--SRN', default='', help="path to SRN (to continue training)")
parser.add_argument('--rec_attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--post', default='', type=str)



print(util.GetNowTime())
print('Begin run!!!')
since = time.time()

opt = parser.parse_args()
sys.stdout.flush()

opt.useSR = True

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

data = dataload.DATA_LOADER(opt)
print("Training Samples: ", data.ntrain)

netG = model.MLP_G_rec(opt)
SRN = model._SRN(opt)
netD = model.MLP_D(opt)
SRN.load_state_dict(torch.load(opt.SRN)['state_dict_RecNet'])
SRN.eval()

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

if opt.dataset == 'CUB':
    opt.f_hid = 7000
if opt.dataset == 'FLO':
    opt.f_hid = 7000
if opt.dataset == 'SUN':
    opt.f_hid = 7000
if opt.dataset == 'AWA1':
    opt.f_hid = 3072
if opt.dataset == 'APY':
    opt.f_hid = 6144

exp_info = '{}'.format(opt.dataset)

exp_params = 'Ablation_B'

out_dir  = 'out/{:s}'.format(exp_info)
out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(out_subdir):
    os.mkdir(out_subdir)
#log_loss_dir = out_subdir + '/log_SRGAN_loss_{:s}.txt'.format(exp_info)
if opt.gzsl:
    log_acc_name = out_subdir + '/log_Ablation_A_acc_GZSL_{:s}_'.format(exp_info)
else:
    log_acc_name = out_subdir + '/log_Ablation_A_acc_ZSL_{:s}_'.format(exp_info)
log_acc_name = log_acc_name + str(opt.rec_attSize) + '.txt'
#with open(log_loss_dir, 'w') as f:
    #f.write('Training Start:')
log_acc_log = util.Logger(log_acc_name)

netS = model.MLP_V2S(opt)
netS2 = model.MLP_V2RS(opt)

cls_criterion = nn.NLLLoss()
reg_criterion = nn.MSELoss()
cnp_criterion = nn.CrossEntropyLoss()
BCELoss = torch.nn.BCELoss(size_average=False)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)


input_res2 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att2 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label2 = torch.LongTensor(opt.batch_size)

input_res3 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att3 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label3 = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netS.cuda()
    SRN.cuda()
    netS2.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    reg_criterion.cuda()
    cnp_criterion.cuda()
    BCELoss.cuda()
    
    input_label = input_label.cuda()

    input_res2 = input_res2.cuda()
    input_att2 = input_att2.cuda()
    input_label2 = input_label2.cuda()

    input_res3 = input_res3.cuda()
    input_att3 = input_att3.cuda()
    input_label3 = input_label3.cuda()

#global variable
d_lr = opt.lr
g_lr = opt.lr
e_lr = opt.lr
s1_lr = 1e-4
s2_lr = 1e-4
if opt.new_lr == 1:
    d_lr = 1e-3
    g_lr = 1e-4

optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(opt.beta1, 0.999))
optimizerS = optim.Adam(netS.parameters(), lr=s1_lr, betas=(opt.beta1, 0.999))
optimizerS2 = optim.Adam(netS2.parameters(), lr=s2_lr, betas=(opt.beta1, 0.999))



pretrain_cls = classifier.CLASSIFIER(data, data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 4096,
                                     opt.pretrain_classifier)
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False
pretrain_cls.model.eval()

def get_negative_samples(Y:list):
    Yp = []
    for y in Y:
        yy = y
        while yy == y:
            yy = np.random.choice(list(data.seenclasses), 1)
        Yp.append(yy[0])
    return Yp

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
def sample2():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res2.copy_(batch_feature)
    input_att2.copy_(batch_att)
    input_label2.copy_(util.map_label(batch_label, data.seenclasses))
def sample3():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res3.copy_(batch_feature)
    input_att3.copy_(batch_att)
    input_label3.copy_(util.map_label(batch_label, data.seenclasses))

def vae_loss_seen_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

def save_Separated_model(opt,competitor,best_acc):
    if opt.gzsl:
        competitor_acc=competitor.H_cls
        competitor_seen_acc = competitor.seen_cls
        competitor_unseen_acc = competitor.unseen_cls
        if competitor_acc>best_acc:
            best_acc = competitor_acc
            files2removeGZSL = glob.glob(out_subdir + '/'+'_Best_model_GZSL_*')
            for _i in files2removeGZSL:
                os.remove(_i)
            torch.save({'state_dict_G': netG.state_dict(),'state_dict_GZSL_classifier': competitor.model.state_dict()}, out_subdir + '/'+'_Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(competitor_acc*100,competitor_seen_acc*100,competitor_unseen_acc*100))
        return best_acc
    else:
        competitor_acc=competitor.cls_acc
        if competitor_acc>best_acc:
            best_acc = competitor_acc
            files2removeZSL = glob.glob(out_subdir+'/'+'Best_model_ZSL_*')
            for _i in files2removeZSL:
                os.remove(_i)
            torch.save({'state_dict_G': netG.state_dict(),'state_dict_ZSL_classifier': competitor.model.state_dict()}, out_subdir +'/'+'Best_model_ZSL_Acc_{:.2f}.tar'.format(competitor_acc*100))
        return best_acc


def generate_syn_feature(netG, SRN, classes, attribute, num):  # 每个类都生成num个
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            input_attv_var = Variable(syn_att)
            syn_noise_var = Variable(syn_noise)
        #print(syn_noise_var.shape)
        #print(input_attv_var.shape)
        input_sr = SRN(input_attv_var)
        output = netG(syn_noise_var, input_attv_var, input_sr)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

def generate_syn_feature_with_grad(netG, SRN, classes, attribute, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    with torch.no_grad():
        input_attv_var = Variable(syn_att)
        syn_noise_var = Variable(syn_noise)
    input_sr = SRN(input_attv_var)
    syn_feature = netG(syn_noise_var, input_attv_var, input_sr)
    return syn_feature, syn_label.cpu()

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def train_D():
     # train with realG
    input_resv = Variable(input_res)
    input_attv = Variable(input_att)    

    # train with fakeG
    #noise.normal_(0, 1)
    #noisev = Variable(noise)
    input_sr = SRN(input_attv)
    criticD_real = netD(input_resv)
    criticD_real = criticD_real.mean()
    criticD_real.backward(mone)
    noise.normal_(0, 1)
    z = Variable(noise)
    fake = netG(z, input_attv, input_sr)
    
    fake_norm = fake.data[0].norm()
    sparse_fake = fake.data[0].eq(0).sum()
    criticD_fake = netD(fake.detach())
    criticD_fake = criticD_fake.mean()
    criticD_fake.backward(one)
    # gradient penalty
    gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)

    gradient_penalty.backward()

    Wasserstein_D = criticD_real - criticD_fake
    D_cost = criticD_fake - criticD_real + gradient_penalty # criticD_fake,criticD_real,gradient_penalty have been backward
    return D_cost,Wasserstein_D

# Train Generator and Decoder
def train_G():
    input_attv = Variable(input_att)
    input_sr = SRN(input_attv)
    noise.normal_(0, 1)
    z = Variable(noise)
    
    fake = netG(z, input_attv, input_sr)
    
    criticG_fake = netD(fake)
    criticG_fake = criticG_fake.mean()
    G_cost = -criticG_fake
    c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
    
    unseen_feature, unseen_label = generate_syn_feature_with_grad(netG, SRN, data.unseenclasses, data.attribute, opt.loss_syn_num)
    unseen_attr = Variable(data.attribute[unseen_label].cuda())
    seen_feature, seen_label = generate_syn_feature_with_grad(netG, SRN, data.seenclasses, data.attribute, opt.loss_syn_num)
    seen_attr = Variable(data.attribute[seen_label].cuda())
    r_errG_seen = reg_criterion(netS(seen_feature), seen_attr)
    r_errG_unseen = reg_criterion(netS(unseen_feature), unseen_attr)
    
    unseen_attr2 = SRN(Variable(data.attribute[unseen_label].cuda()))
    seen_attr2 = SRN(Variable(data.attribute[seen_label].cuda()))
    r_errG_seen2 = reg_criterion(netS2(seen_feature), seen_attr2)
    r_errG_unseen2 = reg_criterion(netS2(unseen_feature), unseen_attr2)
    
    errG = G_cost + opt.cls_weight * c_errG + opt.cyc_seen_weight * r_errG_seen + opt.cyc_unseen_weight * r_errG_unseen + opt.cyc_seen_weight2 * r_errG_seen2 + opt.cyc_unseen_weight2 * r_errG_unseen2
    errG.backward()

    return G_cost
    

def val_ZSL(syn_unseen_feature, syn_unseen_label, best_acc, log_acc_log):
    with torch.no_grad():
        syn_unseen_feature_var = Variable(syn_unseen_feature.cuda())
    
    fake_syn_unseen_attr = netS(syn_unseen_feature_var)
    v2s = soft_cls.Visual_to_semantic(opt, log_acc_log, fake_syn_unseen_attr.data.cpu(), syn_unseen_label, data, data.unseenclasses.size(0), generalized=False)
    opt.zsl_unseen_outputS = v2s.output

    fake_syn_unseen_attr2 = netS2(syn_unseen_feature_var)
    v2sr = soft_cls.Visual_to_RecSemantic(opt, log_acc_log, fake_syn_unseen_attr2.data.cpu(), syn_unseen_label, data, data.unseenclasses.size(0), _lr=opt.softRS_lr, generalized=False)
    opt.zsl_unseen_outputRS = v2sr.output
    
    cls = classifier2forSR.CLASSIFIER(opt, syn_unseen_feature, util.map_label(syn_unseen_label, data.unseenclasses), \
                                      data, data.unseenclasses.size(0), _beta1=0.5, _nepoch=25, generalized=False)
    #save models
    best_acc = save_Separated_model(opt,cls,best_acc)

    #print log
    log_text = 'Visual Softmax: {:.2f}%, bestAcc: {:.2f}%'.format(cls.cls_acc*100, best_acc*100)
    print(log_text)
    log_acc_log.write(log_text+'\n')
    
    return best_acc

def val_GZSL(syn_unseen_feature, syn_unseen_label, best_acc, log_acc_log):
    train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
    train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
    with torch.no_grad():
        train_X_var = Variable(train_X.cuda())
    nclass = opt.nclass_all    
    
    cls = classifier2.CLASSIFIER(opt, train_X_var, train_Y, data, nclass, _beta1=0.5, _nepoch=25, generalized=True)
    #save models
    best_acc = save_Separated_model(opt,cls,best_acc)
    
    #print log
    log_text = 'GZSL Visual Softmax: Seen Acc: {:.2f}%, Unseen Acc: {:.2f}%, H Acc: {:.2f}%, bestAcc: {:.2f}%'.format(cls.seen_cls * 100,cls.unseen_cls * 100,cls.H_cls * 100, best_acc*100)
    print(log_text)
    log_acc_log.write(log_text+'\n')

    return best_acc

#pretrain netS and netS2
netS.train()
for epoch in range(50):
    for i in range(0, data.ntrain, opt.batch_size):
        optimizerS.zero_grad()
        sample()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)
        pred = netS(input_resv)
        loss = reg_criterion(pred, input_attv)
        loss.backward()
        optimizerS.step()
    print(epoch)
    print(loss)
for p in netS.parameters():
    p.requires_grad = False
netS.eval()

netS2.train()
for epoch in range(100):
    for i in range(0, data.ntrain, opt.batch_size):
        optimizerS2.zero_grad()
        sample()
        input_resv = Variable(input_res)
        input_sr = SRN(Variable(input_att))
        pred = netS2(input_resv)
        loss = reg_criterion(pred, input_sr)
        loss.backward()
        optimizerS2.step()
    print(epoch)
    print(loss)
for p in netS2.parameters():
    p.requires_grad = False
netS2.eval()

#pretrain a cls model
pretrain_cls = classifier.CLASSIFIER(data, data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 4096,
                                     opt.pretrain_classifier)
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False
pretrain_cls.model.eval()

with torch.no_grad():
    test_seen_feature_var = Variable(data.test_seen_feature.cuda())
    test_unseen_feature_var = Variable(data.test_unseen_feature.cuda())

if opt.gzsl:
    opt.gzsl_unseen_output, opt.gzsl_seen_output, opt.gzsl_unseen_output2, opt.gzsl_seen_output2 = util.getTestAllAcc_withSR(data,netS,netS2,SRN)
    opt.fake_test_seen_attr2 = netS2(test_seen_feature_var).data
    opt.fake_test_unseen_attr2 = netS2(test_unseen_feature_var).data
    opt.fake_test_seen_attr = netS(test_seen_feature_var).data
    opt.fake_test_unseen_attr = netS(test_unseen_feature_var).data
else:
    opt.zsl_unseen_output, opt.zsl_unseen_output2 = util.getTestUnseenAcc_withSR(data,netS,netS2,SRN)
    opt.fake_test_attr2 = netS2(test_unseen_feature_var).data
    opt.fake_test_attr = netS(test_unseen_feature_var).data
    

#the main code of DCRGAN training
best_acc = 0

for epoch in range(opt.nepoch):
    log_text = 'EP[%d/%d]****************************************************************************************************************' % (epoch, opt.nepoch)
    print(log_text)
    log_acc_log.write(log_text+'\n')
    
    #training stage
    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False
        
        #train D
        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            D_cost,Wasserstein_D = train_D()
            optimizerD.step()

        #train G
        for p in netG.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        G_cost = train_G()
        optimizerG.step()
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f'% (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item()))
    #validation stage
    netG.eval()
    SRN.eval()
    syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, SRN, data.unseenclasses, data.attribute,opt.cls_syn_num)  # 1500x2048
    if opt.gzsl:
        best_acc = val_GZSL(syn_unseen_feature, syn_unseen_label, best_acc, log_acc_log)
    else:
        best_acc = val_ZSL(syn_unseen_feature, syn_unseen_label, best_acc, log_acc_log)
    sys.stdout.flush()
    netG.train()
    SRN.train()

time_elapsed = time.time() - since
print('End run!!!')
print('Time Elapsed: {}'.format(time_elapsed))
print(util.GetNowTime())
