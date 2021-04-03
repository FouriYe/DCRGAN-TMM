import torch.nn as nn
import torch
import torch.nn.functional as F


class _param:
    def __init__(self,opt):
        self.rdc_text_dim = 1000
        self.z_dim = opt.z_dim
        self.h_dim = 4096

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        return self.fc2(h)

class MLP_D_rec(nn.Module):
    def __init__(self, opt): 
        super(MLP_D_rec, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize + opt.rec_attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self, x, att, rec_att):
        h = torch.cat((x, att, rec_att), 1) 
        h = self.lrelu(self.fc1(h))
        return self.fc2(h)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_G_rec(nn.Module):
    def __init__(self, opt):
        super(MLP_G_rec, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.rec_attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att, rec_att):
        h = torch.cat((noise, att, rec_att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_V2S(nn.Module):
    def __init__(self, opt):
        super(MLP_V2S, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.f_hid)
        self.fc2 = nn.Linear(opt.f_hid, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.lrelu = nn.ReLU(True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.relu(self.fc2(h))
        # h = self.fc2(h)
        return h

class MLP_V2RS(nn.Module):
    def __init__(self, opt):
        super(MLP_V2RS, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.f_hid)
        self.fc2 = nn.Linear(opt.f_hid, opt.rec_attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.lrelu = nn.ReLU(True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        # h = self.relu(self.fc2(h))
        h = self.fc2(h)
        return h


class MLP_R(nn.Module):
    def __init__(self,opt):
        super(MLP_R, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.attSize)

    def forward(self, x):
        return self.fc1(x)

class _SRN(nn.Module):
    def __init__(self, opt):
        super(_SRN, self).__init__()
        self.h_dim = opt.attSize*2

        self.main = nn.Sequential(nn.Linear(opt.attSize, self.h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.h_dim, opt.rec_attSize),
                                  nn.Sigmoid())
    def forward(self, input):
        output = self.main(input)
        return output

class _netMetric(nn.Module):
    def __init__(self, opt):
        super(_netMetric, self).__init__()
        if opt.feat=='att+vis':
            in_dim=opt.resSize+opt.attSize
        else:
            in_dim=opt.resSize
        self.h_dim = in_dim*2
        self.main = nn.Sequential(nn.Linear(in_dim, self.h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.h_dim, opt.rec_attSize))
    def forward(self, input):
        output = self.main(input)
        return output


class _seen2unseenM(nn.Module):
    def __init__(self, opt):
        super(_seen2unseenM, self).__init__()
        self.w = nn.Linear(opt.train_cls_num,opt.test_cls_num, bias=False)
    def forward(self, input):
        rec_att = self.w(input)#w(num_useen,num_seen), input_placeholder(num_seen,att_size), pred_y(num_useen,att_size)
        return rec_att