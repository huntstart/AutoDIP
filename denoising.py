# -*- coding: utf-8 -*-
#
from __future__ import print_function
import matplotlib
from torch.version import cuda
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#%matplotlib inline

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   #要求运行时设备查询按照 PCI_BUS_ID 的顺序索引使得 设备ID=物理ID
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
from models import *
from torchvision import transforms
import  torch.utils.data.dataset as DataSet
import torch.utils.data.dataloader as DataLoader
import torch
import torch.optim

from skimage.measure import compare_psnr  # @UnresolvedImport
from utils.denoising_utils import *

import ImageDataset as ID
import sys
import random


# 打印修改时间
print("9.16-19:52")


#设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(5)

cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    
#torch.cuda.set_device(0)

def rdPrint(anyThing,path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a')
    sys.stdout = f
    
    print(anyThing)
    
    sys.stdout = orig_stdout
    f.close()


imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
pic_nub=100

checkfolder = '../denoise-set1'  # 设置各张图片放置out文件目录
print(checkfolder)
os.system('mkdir -p {}/script'.format(checkfolder))
os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
imgDS = ID.ImageDataset(listFile='../data/denoise.list',StartAt=0,Nub=pic_nub)
#使用数据集
for imgIndex in range(0,imgDS.__len__()):
    img,id_img,img_name = imgDS[imgIndex]
    img_pil = crop_image(get_image(img, imsize)[0], d=32)
    
    img_np = pil_to_np(img_pil)

    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)


    #放代码目录,以id_img为各图片文件名，对应id建文件夹
    os.system('mkdir -p {}/{}'.format(checkfolder,id_img))
   # os.system('cp -rfp ../script/*.py {}/{}/script'.format(checkfolder,id_img))

    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 2, 6,it="{}/{}/id{}".format(checkfolder,id_img,id_img));
    
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'
    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01
    
    OPTIMIZER='adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99
    
    num_iter = 6000
    input_depth = 32
    figsize = 4
    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)
   
    
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()            
    
    mse = torch.nn.MSELoss().type(dtype)
    
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    net_input_saved = net_input.detach().clone()#噪声副本
    noise = net_input.detach().clone()#噪声
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

#     pseudo_noise = img_noisy_torch.detach().clone() % it should be different randn noise. bearzhou 20190703
    pseudo_noise = torch.randn(img_noisy_torch.size())
    pseudo_noise = pseudo_noise.normal_()/25
    print('noise:1/25')
    pseudo_noise_np=torch_to_np(pseudo_noise)
    pseudo_noise = pseudo_noise.type(dtype)
    

    plot_image_grid([img_noisy_np, pseudo_noise_np+0.5, pseudo_noise_np + img_noisy_np], 3,6,it="{}/{}/id{}nimg-p-pnimg".format(checkfolder,id_img,id_img));
    error_last = 0
    i = 0
    flog = 0
    def closure():

        global i, out_avg, psrn_noisy_last, last_net, net_input, pseudo_noise,flog,error_last

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        net_input = net_input.type(dtype)

        out = net(net_input)

        total_loss = mse(out, img_noisy_torch+pseudo_noise)
        total_loss.backward()

        error = torch.mean(out*pseudo_noise)
        rdPrint('%i\t%.7f'%(i,error),'{}/{}/pseudoErrorAtEpoch.log'.format(checkfolder,id_img))

        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        #保存峰值信噪比
        rdPrint('%i\t%.7f'%(i,psrn_gt),'{}/{}/psrn_{}.log'.format(checkfolder,id_img,id_img))
        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f ' % (i, total_loss.item(), psrn_noisy, psrn_gt), '\r', end='')

        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=6, nrow=1,it="{}/{}/id{}-{}".format(checkfolder,id_img,id_img,i) )

        flog = 0
        if i>1000:
            if error_last - error > 0.0002:
                flog = 1
                return total_loss*0,flog
            else:
                error_last = error

        i = i+1
        return total_loss,flog




    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR,num_iter)

    out_np = torch_to_np(net(net_input))
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], nrow=2,factor=6,it="{}/{}/id{}-out".format(checkfolder,id_img,id_img) );



