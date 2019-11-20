# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#%matplotlib inline

import scipy.ndimage
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   #要求运行时设备查询按照 PCI_BUS_ID 的顺序索引使得 设备ID=物理ID
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from models import *
import sys
import random
import torch
import torch.optim

from skimage.measure import compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *
import ImageDataset as ID

cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
#torch.cuda.set_device(0)

#日志函数
def rdPrint(anyThing,path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a')
    sys.stdout = f
    
    print(anyThing)
    
    sys.stdout = orig_stdout
    f.close()


imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True
pic_nub=100
#修改时间
print("11131850")


checkfolder = '../sr-es2-set14-4/'  # 设置各张图片放置out文件目录
#os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
os.system('mkdir -p {}/script'.format(checkfolder))
os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
# Starts here
imgDS = ID.ImageDataset(listFile='../data/set14.list',StartAt=0,Nub=pic_nub)
#使用数据集
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
    imgs = load_LR_HR_imgs_sr(img , imsize, factor, enforse_div32)
    os.system('mkdir -p {}/{}'.format(checkfolder,id_img))
    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

    if PLOT:
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12,it="{}/{}/id{}HRbicSharpNearesr".format(checkfolder,id_img,id_img))
        print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                            compare_psnr(imgs['HR_np'], imgs['bicubic_np']),
                                            compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

    input_depth = 32

    INPUT =     'noise'
    pad   =     'reflection'
    OPT_OVER =  'net'
    KERNEL_TYPE='lanczos2'

    LR = 0.01
    tv_weight = 0.0

    OPTIMIZER = 'adam'

    if factor == 4:
        num_iter = 5000
        reg_noise_std = 0.03
    elif factor == 8:
        num_iter = 5000
        reg_noise_std = 0.05
    else:
        assert False, 'We did not experiment with other factors'

    net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

    NET_TYPE = 'skip' # UNet, ResNet
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
    img_HR_var = np_to_torch(imgs['HR_np']).type(dtype)
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

    #伪噪声
    pseudo_noise = torch.randn(img_HR_var.size())
    pseudo_noise = pseudo_noise.normal_()/25 #BearZhou: was 25, use smaller noise making the problem harder.

    pseudo_noise_np=torch_to_np(pseudo_noise)

    pseudo_noise = pseudo_noise.type(dtype)



    error_last = 0
    flog = 0
    def closure():
        global i, net_input,flog,error_last

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        pseudo_noise_downsample = downsampler(pseudo_noise)


        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var+pseudo_noise_downsample)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()

        #计算error和psrn并保存
        error = torch.mean(out_HR*pseudo_noise)
        error2 = torch.mean(out_LR*pseudo_noise_downsample)
        rdPrint('%i\t%.7f'%(i,error),'{}/{}/pseudoErrorAtEpoch.log'.format(checkfolder,id_img))
        rdPrint('%i\t%.7f'%(i,error2),'{}/{}/pseudoErrorAtEpoch2.log'.format(checkfolder,id_img))
    #保存峰值信噪比


        # Log
        psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
     #   psrn = compare_psnr(imgs['HR_np'], torch_to_np(out_LR))
        psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
        print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')


        rdPrint('%i\t%.7f'%(i,psnr_LR),'{}/{}/psrn_LR.log'.format(checkfolder,id_img))
      #  rdPrint('%i\t%.7f'%(i,psnr),'./psrn.log')
        rdPrint('%i\t%.7f'%(i,psnr_HR),'{}/{}/psrn_HR.log'.format(checkfolder,id_img))
        # History
        psnr_history.append([psnr_LR, psnr_HR])

        
        if PLOT and i % 10 == 0:
            out_HR_np = torch_to_np(out_HR)
            plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3,it= "{}/{}/id{}-{}".format(checkfolder,id_img,id_img,i))
        flog = 0
        if i > 1000:
            if error_last - error > 0.0001:
                flog = 1
                return total_loss * 0, 1
            else:
                error_last = error

        i += 1

        return total_loss,flog

    psnr_history = []
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
    result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])


    plot_image_grid([imgs['HR_np'],
                     imgs['bicubic_np'],
                     out_HR_np], factor=4, nrow=1,it="{}/{}/id{}-out".format(checkfolder,id_img,id_img));

