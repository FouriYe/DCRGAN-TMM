# encoding: utf-8
#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from torch.autograd import Variable
import torch.nn as nn
import time

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename, "w")
        f.write('Training Start:\n')
        f.close()

    def write(self, message):
        f = open(self.filename, "a")
        f.write(message)
        f.close()


def getTrainSeenAcc(data,netS):
    with torch.no_grad():
        train_feature_var = Variable(data.train_feature.cuda())
    fake_train_attr = netS(train_feature_var)
    dist = pairwise_distances(fake_train_attr.data, data.attribute[data.seenclasses].cuda())  # range 150
    pred_idx = torch.min(dist, 1)[1]
    pred = data.seenclasses[pred_idx.cpu()]
    acc = sum(pred == data.train_label) / data.train_label.size()[0]

def getTrainSeenAcc_withSR(data,netS,netS2,SRN):
    with torch.no_grad():
        train_feature_var = Variable(data.train_feature.cuda())
    fake_train_attr = netS(train_feature_var)
    dist = pairwise_distances(fake_train_attr.data, data.attribute[data.seenclasses].cuda())  # range 150
    pred_idx = torch.min(dist, 1)[1]
    pred = data.seenclasses[pred_idx.cpu()]
    acc = sum(pred == data.train_label) / data.train_label.size()[0]
    
    fake_train_attr_RS = netS2(train_feature_var)
    dist_RS = pairwise_distances(fake_train_attr_RS.data, SRN(data.attribute[data.seenclasses].cuda()) )  # range 150
    pred_idx_RS = torch.min(dist_RS, 1)[1]
    pred_RS = data.seenclasses[pred_idx_RS.cpu()]
    acc_RS = sum(pred_RS == data.train_label) / data.train_label.size()[0]
def getTestUnseenAcc(data,netS):
    logsoftmax = nn.LogSoftmax(dim=1)
    logsoftmax.cuda()
    with torch.no_grad():
        test_unseen_feature_var = Variable(data.test_unseen_feature.cuda())
    fake_unseen_attr = netS(test_unseen_feature_var)
    dist = pairwise_distances(fake_unseen_attr.data, data.attribute[data.unseenclasses].cuda())  # range 50
    pred_idx = torch.min(dist, 1)[1]  # relative pred
    pred = data.unseenclasses[pred_idx.cpu()]  # map relative pred to absolute pred
    acc = sum(pred == data.test_unseen_label) / data.test_unseen_label.size()[0]
    
    return logsoftmax(Variable(dist.cuda())).data

def getTestUnseenAcc_withSR(data,netS,netS2,SRN):
    logsoftmax = nn.LogSoftmax(dim=1)
    logsoftmax.cuda()
    with torch.no_grad():
        test_unseen_feature_var = Variable(data.test_unseen_feature.cuda())
    fake_unseen_attr = netS(test_unseen_feature_var)
    dist = pairwise_distances(fake_unseen_attr.data, data.attribute[data.unseenclasses].cuda())  # range 50
    pred_idx = torch.min(dist, 1)[1]  # relative pred
    pred = data.unseenclasses[pred_idx.cpu()]  # map relative pred to absolute pred
    acc = sum(pred == data.test_unseen_label) / data.test_unseen_label.size()[0]
    
    fake_unseen_attr_RS = netS2(test_unseen_feature_var)
    dist_RS = pairwise_distances(fake_unseen_attr_RS.data, SRN(data.attribute[data.unseenclasses].cuda()) )  # range 50
    pred_idx_RS = torch.min(dist_RS, 1)[1]  # relative pred
    pred_RS = data.unseenclasses[pred_idx_RS.cpu()]  # map relative pred to absolute pred
    acc_RS = sum(pred_RS == data.test_unseen_label) / data.test_unseen_label.size()[0]
    
    return logsoftmax(Variable(dist.cuda())).data, logsoftmax(Variable(dist_RS.cuda())).data

def getTestAllAcc(data,netS):
    logsoftmax = nn.LogSoftmax(dim=1)
    logsoftmax.cuda()
    with torch.no_grad():
        test_unseen_feature_var = Variable(data.test_unseen_feature.cuda())
        test_seen_feature_var = Variable(data.test_seen_feature.cuda())
    fake_unseen_attr = netS(test_unseen_feature_var)
    dist1 = pairwise_distances(fake_unseen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist1, 1)[1]  # absolute pred
    acc_unseen = sum(pred_idx.cpu() == data.test_unseen_label) / data.test_unseen_label.size()[0]

    fake_seen_attr = netS(test_seen_feature_var)
    dist2 = pairwise_distances(fake_seen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist2, 1)[1]  # absolute pred
    acc_seen = sum(pred_idx.cpu() == data.test_seen_label) / data.test_seen_label.size()[0]

    if (acc_seen == 0) or (acc_unseen == 0):
        H = 0
    else:
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    
    return logsoftmax(Variable(dist1.cuda())).data, logsoftmax(Variable(dist2.cuda())).data

def getTestAllAcc_withSR(data,netS,netS2,SRN):
    logsoftmax = nn.LogSoftmax(dim=1)
    logsoftmax.cuda()
    with torch.no_grad():
        test_unseen_feature_var = Variable(data.test_unseen_feature.cuda())
        test_seen_feature_var = Variable(data.test_seen_feature.cuda())
    fake_unseen_attr = netS(test_unseen_feature_var)
    dist1 = pairwise_distances(fake_unseen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist1, 1)[1]  # absolute pred
    acc_unseen = sum(pred_idx.cpu() == data.test_unseen_label) / data.test_unseen_label.size()[0]

    fake_seen_attr = netS(test_seen_feature_var)
    dist2 = pairwise_distances(fake_seen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist2, 1)[1]  # absolute pred
    acc_seen = sum(pred_idx.cpu() == data.test_seen_label) / data.test_seen_label.size()[0]

    if (acc_seen == 0) or (acc_unseen == 0):
        H = 0
    else:
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    
    fake_unseen_attr_RS = netS2(test_unseen_feature_var)
    SR = SRN(data.attribute.cuda())
    dist1_RS = pairwise_distances(fake_unseen_attr_RS.data, SR)  # range 200
    pred_idx_RS = torch.min(dist1_RS, 1)[1]  # absolute pred
    acc_unseen_RS = sum(pred_idx_RS.cpu() == data.test_unseen_label) / data.test_unseen_label.size()[0]

    fake_seen_attr_RS = netS2(test_seen_feature_var)
    dist2_RS = pairwise_distances(fake_seen_attr_RS.data, SR)  # range 200
    pred_idx_RS = torch.min(dist2_RS, 1)[1]  # absolute pred
    acc_seen_RS = sum(pred_idx_RS.cpu() == data.test_seen_label) / data.test_seen_label.size()[0]

    if (acc_seen_RS == 0) or (acc_unseen_RS == 0):
        H_RS = 0
    else:
        H_RS = 2 * acc_seen_RS * acc_unseen_RS / (acc_seen_RS + acc_unseen_RS)
    
    return logsoftmax(Variable(dist1.cuda())).data, logsoftmax(Variable(dist2.cuda())).data,logsoftmax(Variable(dist1_RS.cuda())).data, logsoftmax(Variable(dist2_RS.cuda())).data

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)