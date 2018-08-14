import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import data_loader
from torch.autograd import Variable
from model import _G_xvz, _G_vzx, _D_xvs
from itertools import *
import pdb

dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str, default="./list.txt")
parser.add_argument("-ns", "--nsnapshot", type=int, default=700)
parser.add_argument("-b", "--batch_size", type=int, default=64) # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
parser.add_argument("-m" , "--momentum", type=float, default=0.) # 0.5
parser.add_argument("-m2", "--momentum2", type=float, default=0.9) # 0.999
parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')

# Initialize networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_xvz = _G_xvz()
G_vzx = _G_vzx()
D_xvs = _D_xvs()

G_xvz.apply(weights_init)
G_vzx.apply(weights_init)
D_xvs.apply(weights_init)


train_list = args.data_list
train_loader = torch.utils.data.DataLoader(
    data_loader.ImageList( train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x-y), 1))

v_siz = 9
z_siz = 128-v_siz
x1 = torch.FloatTensor(args.batch_size, 3, 128, 128)
x2 = torch.FloatTensor(args.batch_size, 3, 128, 128)
v1 = torch.FloatTensor(args.batch_size, v_siz)
v2 = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_xvz = torch.nn.DataParallel(G_xvz).cuda()
    G_vzx = torch.nn.DataParallel(G_vzx).cuda()
    D_xvs = torch.nn.DataParallel(D_xvs).cuda()
    x1 = x1.cuda()
    x2 = x2.cuda()
    v1 = v1.cuda()
    v2 = v2.cuda()
    z = z.cuda()

x1 = Variable(x1)
x2 = Variable(x2)
v1 = Variable(v1)
v2 = Variable(v2)
z = Variable(z)

def load_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path,name))
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)

#load_model(G_xvz, args.modelf, 'netG_xvz_epoch_24_699.pth')
#load_model(G_vzx, args.modelf, 'netG_vzx_epoch_24_699.pth')
#load_model(D_xvs, args.modelf, 'netD_xvs_epoch_24_699.pth')

lr = args.learning_rate
ourBetas = [args.momentum, args.momentum2]
batch_size = args.batch_size
snapshot = args.nsnapshot
start_time = time.time()

G_xvz_solver = optim.Adam(G_xvz.parameters(), lr = lr, betas=ourBetas)
G_vzx_solver = optim.Adam(G_vzx.parameters(), lr = lr, betas=ourBetas)
D_xvs_solver = optim.Adam(D_xvs.parameters(), lr = lr, betas=ourBetas)

cudnn.benchmark = True

crossEntropyLoss = nn.CrossEntropyLoss().cuda()

for epoch in range(args.epochs):
    for i, (view1, view2, data1, data2) in enumerate(train_loader):
        # our framework:
        # path 1: (v, z)-->G_vzx-->x_bar--> D_xvs( (v,x_bar), (v,x) )
        # This path to make sure G_vzx can generate good quality images with any random input
        # path 2: x-->G_xvz-->(v_bar, z_bar)-->G_vzx-->x_bar_bar--> D_xvs( (v,x_bar_bar), (v,x) ) + L1_loss(x_bar_bar, x)
        # This path to make sure G_xvz is the reverse of G_vzx
        eps = random.uniform(0, 1)
        tmp = random.uniform(0, 1)
        reconstruct_fake = False
        if tmp < 0.5:
            reconstruct_fake = True

        D_xvs.zero_grad()
        G_xvz.zero_grad()
        G_vzx.zero_grad()

        img1 = data1
        img2 = data2

        # get x-->real image v--> view and z-->random vector
        x1.data.resize_(img1.size()).copy_(img1)
        x2.data.resize_(img2.size()).copy_(img2)
        v1.data.zero_()
        v2.data.zero_()

        for d in range(view1.size(0)):
            v1.data[d][view1[d]] = 1

        for d in range(view2.size(0)):
            v2.data[d][view2[d]] = 1

        z.data.uniform_(-1, 1) # random z
        
        targetNP = v1.cpu().data.numpy()
        idxs = np.where(targetNP>0)[1]
        tmp = torch.LongTensor(idxs)
        vv1 = Variable(tmp).cuda() # v1 target

        targetNP = v2.cpu().data.numpy()
        idxs = np.where(targetNP>0)[1]
        tmp = torch.LongTensor(idxs)
        vv2 = Variable(tmp).cuda() # v2 target
        
        ## path 1: (v, z)-->G_vzx-->x_bar--> D_xvs( (v,x_bar), (v,x_real) )
        # path 1, update D_xvs
        x_bar = G_vzx(v1, z) # random z to generate img x_bar

        x_hat = eps*x1.data + (1-eps)*x_bar.data # interpolation of x_bar and x1
        x_hat = Variable(x_hat, requires_grad=True)
        D_x_hat_v, D_x_hat_s = D_xvs(x_hat)

        grads = autograd.grad(outputs = D_x_hat_s,
                              inputs = x_hat,
                              grad_outputs = torch.ones(D_x_hat_s.size()).cuda(),
                              retain_graph = True,
                              create_graph = True,
                              only_inputs = True)[0]
        grad_norm = grads.pow(2).sum().sqrt()
        gp_loss = torch.mean((grad_norm - 1) ** 2) # gradient with v1

        x_bar_loss_v, x_bar_loss_s = D_xvs(x_bar.detach()) # score of x_bar
        x_bar_loss_s = x_bar_loss_s.mean()

        x_loss_v, x_loss_s = D_xvs(x1) # score of x1
        x_loss_s = x_loss_s.mean()

        v_loss_x = crossEntropyLoss(x_loss_v, vv1) # ACGAN loss of x1(v1)
        
        d_xvs_loss = x_bar_loss_s - x_loss_s + 10. * gp_loss + v_loss_x # x1 real sample, x_bar fake sample
        d_xvs_loss.backward()
        D_xvs_solver.step()

        # path 1, update G_vzx
        D_xvs.zero_grad()
        G_xvz.zero_grad()
        G_vzx.zero_grad()

        x_bar_loss_v, x_bar_loss_s = D_xvs(x_bar) # score of x_bar
        x_bar_loss_s = x_bar_loss_s.mean()

        v_loss_x_bar = crossEntropyLoss(x_bar_loss_v, vv1) # ACGAN loss of x_bar(v1)

        g_vzx_loss = -x_bar_loss_s + v_loss_x_bar
        g_vzx_loss.backward()
        G_vzx_solver.step()

        ## path 2: x-->G_xvz-->(v_bar, z_bar)-->G_vzx-->x_bar_bar--> D_xvs( (v,x_bar_bar), (v,x) ) + L1_loss(x_bar_bar, x)
        # path 2, update D_x
        D_xvs.zero_grad()
        G_xvz.zero_grad()
        G_vzx.zero_grad()

        if reconstruct_fake is True:
            v_bar, z_bar = G_xvz(x_bar.detach())
            x_bar_bar = G_vzx(v1, z_bar)
            x_hat = eps*x1.data + (1-eps)*x_bar_bar.data # interpolation of x1 and x_bar_bar
        else:
            v_bar, z_bar = G_xvz(x1) # view invariant part of x1 
            x_bar_bar = G_vzx(v2, z_bar) # x_bar_bar: reconstruction of x2
            x_hat = eps*x2.data + (1-eps)*x_bar_bar.data # interpolation of x2 and x_bar_bar
        
        x_hat = Variable(x_hat, requires_grad=True)
        D_x_hat_v, D_x_hat_s = D_xvs(x_hat)

        grads = autograd.grad(outputs = D_x_hat_s,
                              inputs = x_hat,
                              grad_outputs = torch.ones(D_x_hat_s.size()).cuda(),
                              retain_graph = True,
                              create_graph = True,
                              only_inputs = True)[0]
        grad_norm = grads.pow(2).sum().sqrt()
        gp_loss = torch.mean((grad_norm - 1) ** 2)
        
        x_loss_v, x_loss_s = D_xvs(x2)
        x_loss_s = x_loss_s.mean()
        x_bar_bar_loss_v, x_bar_bar_loss_s = D_xvs(x_bar_bar.detach()) # x_bar_bar score
        x_bar_bar_loss_s = x_bar_bar_loss_s.mean()
        
        v_loss_x = crossEntropyLoss(x_loss_v, vv2) # ACGAN loss of x2(v2)

        d_x_loss = x_bar_bar_loss_s - x_loss_s + 10. * gp_loss + v_loss_x
        d_x_loss.backward()
        D_xvs_solver.step()

        # 2st path, update G_xvz
        x_bar_bar_loss_v, x_bar_bar_loss_s = D_xvs(x_bar_bar) # x_bar_bar score
        x_bar_bar_loss_s = x_bar_bar_loss_s.mean()
        
        if reconstruct_fake is True:
            x_l1_loss = L1_loss(x_bar_bar, x_bar.detach())
            v_loss_x_bar_bar = crossEntropyLoss(x_bar_bar_loss_v, vv1) # ACGAN loss of x_bar_bar(v1)
        else:
            x_l1_loss = L1_loss(x_bar_bar, x2) # L1 loss between x_bar_bar and x2
            v_loss_x_bar_bar = crossEntropyLoss(x_bar_bar_loss_v, vv2) # ACGAN loss of x_bar_bar(v2)
        
        v_loss_x = crossEntropyLoss(v_bar, vv1)
        g_loss = -x_bar_bar_loss_s + 4*x_l1_loss + v_loss_x_bar_bar + 0.01*v_loss_x 
        g_loss.backward()


        if reconstruct_fake is False:
            G_vzx_solver.step()
        
        G_xvz_solver.step()

        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "
              "loss_D_vx: %.4f, loss_D_x: %.4f, loss_G: %.4f"
              % (epoch, i, len(data1), time.time() - start_time,
                 d_xvs_loss.data[0], d_x_loss.data[0], g_loss.data[0]))
        if i % snapshot == snapshot-1:
            vutils.save_image(x_bar.data,
                              '%s/x_bar_epoch_%03d_%04d.png' % (args.outf, epoch, i),normalize=True)
            vutils.save_image(x_bar_bar.data,
                              '%s/x_bar_bar_epoch_%03d_%04d.png' % (args.outf, epoch, i),normalize=True)
            vutils.save_image(x1.data,
                    '%s/x1_epoch_%03d_%04d.png' % (args.outf, epoch, i),normalize=True)
            vutils.save_image(x2.data,
                    '%s/x2_epoch_%03d_%04d.png' % (args.outf, epoch, i),normalize=True)

            torch.save(G_xvz.state_dict(), '%s/netG_xvz_epoch_%d_%d.pth' % (args.outf, epoch, i))
            torch.save(G_vzx.state_dict(), '%s/netG_vzx_epoch_%d_%d.pth' % (args.outf, epoch, i))
            torch.save(D_xvs.state_dict(), '%s/netD_xvs_epoch_%d_%d.pth' % (args.outf, epoch, i))

