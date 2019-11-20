# -*- coding: utf-8 -*-

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
from models.skip import skip
from skimage.measure import compare_psnr
import torch
import torch.optim
import random
import math
from utils.inpainting_utils import *

import ImageDataset as ID

import sys
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor




PLOT = True
imsize = -1
dim_div_by = 64

cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
#torch.cuda.set_device(0)
def rdPrint(anyThing, path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a')
    sys.stdout = f

    print(anyThing)

    sys.stdout = orig_stdout
    f.close()

#修改时间
print("1104-1708")
NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pic_nub=100
checkfolder = '../inpainting-es1'  # 设置各张图片放置out文件目录
print(checkfolder)
os.system('mkdir -p {}/script'.format(checkfolder))
os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
imgDS = ID.ImageDataset(listFile='../data/inpainting_img.list',StartAt=0,Nub=pic_nub)
maskDs = ID.ImageDataset(listFile='../data/inpainting_mask.list',StartAt=0,Nub=pic_nub)
for imgIndex in range(0,imgDS.__len__()):

    # 设置随机数种子
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    # 设置随机数种子
    setup_seed(5)

    img, id_img,img_name = imgDS[imgIndex]
    os.system('mkdir -p {}/{}'.format(checkfolder,id_img))
    img_pil, img_np = get_image(img, imsize)
    mask_img, mask_id,mask_name = maskDs[imgIndex]
    img_mask_pil, img_mask_np = get_image(mask_img, imsize)

    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil      = crop_image(img_pil,      dim_div_by)

    img_np      = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    img_mask_var = np_to_torch(img_mask_np).type(dtype)
    plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11,it="{}/{}/id{}".format(checkfolder,id_img,id_img));


    pad = 'reflection' # 'zero'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'
    if 'vase' in img_name:
        INPUT = 'noise'
        input_depth = 2
        LR = 0.01
        num_iter = 8000
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03

        net = skip(input_depth, img_np.shape[0],
                   num_channels_down = [128]*5,
                   num_channels_up =   [128]*5,
                   num_channels_skip =  [0]*5,
                   upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    elif 'kate.png' in img_name:

        INPUT = 'noise'
        input_depth = 32
        LR = 0.01
        num_iter = 8000
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03

        net = skip(input_depth, img_np.shape[0],
                   num_channels_down = [32, 64, 128, 128, 128],
                   num_channels_up =   [32, 64, 128, 128, 128],
                   num_channels_skip =    [0, 0, 0, 4, 4],
                   filter_size_up = 3, filter_size_down = 3,
                   upsample_mode='nearest', filter_skip_size=1,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    elif 'library' in img_name:

        INPUT = 'noise'
        input_depth = 1

        num_iter = 8000
        show_every = 50
        figsize = 8
        reg_noise_std = 0.00
        param_noise = True

        if 'skip' in NET_TYPE:

            depth = int(NET_TYPE[-1])
            net = skip(input_depth, img_np.shape[0],
                   num_channels_down = [32, 64, 128, 128, 128, 128][:depth],#
                   num_channels_up =   [32, 64, 128, 128, 128, 128][:depth],#
                   num_channels_skip =    [0, 0, 0, 4, 4, 4][:depth],
                   filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                   upsample_mode='nearest', # downsample_mode='avg',
                   need1x1_up=False,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

            LR = 0.01



        else:
            assert False
    else:
        assert False

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

    # Compute number of parameters
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    img_noisy_torch = img_var * mask_var
    pseudo_noise = torch.randn(img_noisy_torch.size())

    # i为通道数，j为行数，k为列数
    for i in range(pseudo_noise.shape[1]):
        for j in range(pseudo_noise.shape[2]):
            #各行A、w独立存在
            A = random.uniform(1 / 50, 1 / 25)
            w = random.uniform((math.pi) * 20 / 512, (math.pi) * 40 / 512)
            #print('the {} raw A/w is :'.format(j),A,w)
            for k in range(pseudo_noise.shape[3]):
                pseudo_noise[0][i][j][k] = A * math.sin(w * (k+1))

    pseudo_noise = pseudo_noise.type(dtype)
    pseudo_noise = pseudo_noise * mask_var
    pseudo_noise_np = torch_to_np(pseudo_noise)

    img_noisy_np = torch_to_np(img_noisy_torch)
    plot_image_grid([img_noisy_np, pseudo_noise_np+0.5, pseudo_noise_np + img_noisy_np], 3,6,it="{}/{}/id{}nimg-p-pnimg".format(checkfolder,id_img,id_img));

    error_list = []
    psrn_list = []

    #psrn_noisy_last = 0
    last_net= None
    #total_loss_last = 0
    error_last = 0
    i = 0
    count = 0
    flog = 0


    def closure():

        global i,flog,psrn_noisy_last,last_net,total_loss_last,count,error_last,error,error_list,psrn_list

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)


        out = net(net_input)

        total_loss = mse(out* mask_var , img_var * mask_var+pseudo_noise)
        total_loss.backward()

        #psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        error = torch.sum(out * mask_var * pseudo_noise) / torch.sum(mask_var)
        psrn = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        rdPrint('%i\t%.7f' % (i, error), '{}/{}/pseudoErrorAtEpoch.log'.format(checkfolder, id_img))
        rdPrint('%i\t%.7f'%(i,psrn),'{}/{}/psrn_{}.log'.format(checkfolder,id_img,id_img))

        print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')

        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=6, nrow=1,it="{}/{}/id{}-{}".format(checkfolder,id_img,id_img,i) )
        flog = 0
        if i>1000:
            if error_last - error > 0.0001:
                flog = 1
                return total_loss*0,flog
            else:
                error_last = error

        i += 1

        return total_loss,flog


    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = torch_to_np(net(net_input))
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], nrow=2,factor=6,it="{}/{}/id{}-out".format(checkfolder,id_img,id_img) );
