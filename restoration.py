# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#%matplotlib inline

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   #要求运行时设备查询按照 PCI_BUS_ID 的顺序索引使得 设备ID=物理ID
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np

from models import *
import torch
import torch.optim
import  torch.utils.data.dataset as DataSet
import torch.utils.data.dataloader as DataLoader
from skimage.measure import compare_psnr
from utils.denoising_utils import *
from utils.inpainting_utils import *
import ImageDataset as ID
cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
#torch.cuda.set_device(1)

def rdPrint(anyThing,path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a')
    sys.stdout = f
    
    print(anyThing)
    
    sys.stdout = orig_stdout
    f.close()

PLOT = True
imsize=-1
dim_div_by = 64

sigma = 25
sigma_ = sigma/255.
pic_nub=2

#修改时间
print("929-1245")
checkfolder = '../restoration-es5/'  # 设置各张图片放置out文件目录
print(checkfolder)
#os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
os.system('mkdir -p {}/script'.format(checkfolder))
os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))
imgDS = ID.ImageDataset(listFile='../data/restoration.list',StartAt=0,Nub=pic_nub)
for imgIndex in range(0,imgDS.__len__()):
    img, id_img ,img_name = imgDS[imgIndex]
    if id_img==1:
        img = img.convert("RGB")
    img_pil, img_np = get_image(img, imsize)

#    print('img shape: ',img_np,img_np.shape)
#    print('solo channle',img_np[:, :, 0], img_np[:, :, 0].shape)

    #print('img shape: ',img_np,img_np.shape)
    img_np = nn.ReflectionPad2d(1)(np_to_torch(img_np))[0].numpy()

    img_pil = np_to_pil(img_np)

    img_mask = get_bernoulli_mask(img_pil, 0.50)
    img_mask_np = pil_to_np(img_mask)
    
    os.system('mkdir -p {}/{}'.format(checkfolder,id_img))
    img_masked = img_np * img_mask_np

    mask_var = np_to_torch(img_mask_np).type(dtype)

    #伪噪声
    pseudo_noise = torch.randn(mask_var.size())
    pseudo_noise = pseudo_noise.normal_()/25

    # BearZhou: do mask on pseudo_noise
    pseudo_noise = pseudo_noise.type(dtype)
    pseudo_noise = pseudo_noise*mask_var

    pseudo_noise_np=torch_to_np(pseudo_noise)


    plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3,11,it="{}/{}/id{}".format(checkfolder,id_img,id_img));
    plot_image_grid([img_masked, pseudo_noise_np, pseudo_noise_np + img_masked], 3,11,it="{}/{}/id{}nimg-p-pnimg".format(checkfolder,id_img,id_img));



    show_every=100
    figsize=5
    pad = 'reflection' # 'zero'
    INPUT = 'noise'
    input_depth = 32
    OPTIMIZER = 'adam'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'

    LR = 0.001
    num_iter = 10000
    reg_noise_std = 0.03


    NET_TYPE = 'skip'
    net = get_net(input_depth, 'skip', pad, n_channels=3,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_np).type(dtype)

    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype).detach()#此处为restoration2的实验

    #net_input = get_noise(input_depth, INPUT, (img_pil.size[1],img_pil.size[0])).type(dtype).detach()这个效果不大好

    # Init globals
    last_net = None
    psrn_masked_last = 0
    i = 0

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    def closure():

        global i, psrn_masked_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)


        out = net(net_input)
        total_loss = mse(out * mask_var, img_var * mask_var + pseudo_noise)#最开始的
        total_loss.backward()
        OurMean = torch.mean(torch.sum(out * mask_var * pseudo_noise) / torch.sum(mask_var))  # 最开始的
        psrn = compare_psnr(img_np, out.detach().cpu().numpy()[0])



        rdPrint('%i\t%.7f' % (i, OurMean), '{}/{}/pseudoErrorAtEpoch.log'.format(checkfolder, id_img))

        psrn_masked = compare_psnr(img_masked, out.detach().cpu().numpy()[0] * img_mask_np)

        #psrn = compare_psnr(img_np, torch.exp(out).detach().cpu().numpy()[0])

        #保存峰值信噪比
        rdPrint('%i\t%.7f'%(i,psrn),'{}/{}/psrn_{}.log'.format(checkfolder,id_img,id_img))

        print('Iteration %05d    Loss %f PSNR_masked %f PSNR %f' % (i, total_loss.item(), psrn_masked, psrn), '\r',
              end='')

        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1,it="{}/{}/id{}-{}".format(checkfolder,id_img,id_img,i) )

        i += 1


        return total_loss , 0



    # Run
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR=LR, num_iter=num_iter)

    out_np = torch_to_np(net(net_input))

    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13,it="{}/{}/id{}-out".format(checkfolder,id_img,id_img));
