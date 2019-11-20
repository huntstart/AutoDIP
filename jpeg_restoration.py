from __future__ import print_function
import matplotlib.pyplot as plt

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   #要求运行时设备查询按照 PCI_BUS_ID 的顺序索引使得 设备ID=物理ID
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
import ImageDataset as ID
import sys
cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def rdPrint(anyThing, path='./'):
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
pic_nub = 1
# deJPEG

checkfolder = '../es-snail/'  # 设置各张图片放置out文件目录
print(checkfolder)
os.system('mkdir -p {}/script'.format(checkfolder))
os.system('cp -rfp ../script/*.py {}/script'.format(checkfolder))

## denoising
imgDS = ID.ImageDataset(listFile='../data/denoise.list',StartAt=0,Nub=pic_nub)
for j in range(0,3):
	imgIndex=0
#for imgIndex in range(0,imgDS.__len__()):

	img,id_img,img_name = imgDS[imgIndex]
	os.system('mkdir -p {}/{}'.format(checkfolder, id_img))

	if 'snail' in img_name :
		img_noisy_pil = crop_image(get_image(img, imsize)[0], d=32)
		img_noisy_np = pil_to_np(img_noisy_pil)

		# As we don't have ground truth
		img_pil = img_noisy_pil
		img_np = img_noisy_np

		if PLOT:
			plot_image_grid([img_np, img_noisy_np], 2, 6, it="{}/{}/id{}".format(checkfolder, id_img, id_img));

	elif 'GT' in img_name:
		# Add synthetic noise
		img_pil = crop_image(get_image(img, imsize)[0], d=32)
		img_np = pil_to_np(img_pil)

		img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

		if PLOT:
			plot_image_grid([img_np, img_noisy_np], 4, 6);
	else:
		assert False

	INPUT = 'noise'  # 'meshgrid'
	pad = 'reflection'
	OPT_OVER = 'net'  # 'net,input'

	reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
	LR = 0.01

	OPTIMIZER = 'adam'  # 'LBFGS'
	show_every = 100
	exp_weight = 0.99

	if 'snail' in img_name :
		num_iter = 2400
		input_depth = 3
		figsize = 5

		net = skip(
			input_depth, 3,
			num_channels_down=[8, 16, 32, 64, 128],
			num_channels_up=[8, 16, 32, 64, 128],
			num_channels_skip=[0, 0, 0, 4, 4],
			upsample_mode='bilinear',
			need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

		net = net.type(dtype)
	elif 'GT' in img_name:
		num_iter = 3000
		input_depth = 32
		figsize = 4

		net = get_net(input_depth, 'skip', pad,
					  skip_n33d=128,
					  skip_n33u=128,
					  skip_n11=4,
					  num_scales=5,
					  upsample_mode='bilinear').type(dtype)

	else:
		assert False

	net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

	# Compute number of parameters
	s = sum([np.prod(list(p.size())) for p in net.parameters()]);
	print('Number of params: %d' % s)

	# Loss
	mse = torch.nn.MSELoss().type(dtype)

	img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

	net_input_saved = net_input.detach().clone()
	noise = net_input.detach().clone()
	out_avg = None
	last_net = None
	psrn_noisy_last = 0

	i = 0
	pseudo_noise = torch.randn(img_noisy_torch.size())
	pseudo_noise = pseudo_noise.normal_() / 25
	print('noise:1/25')
	pseudo_noise_np = torch_to_np(pseudo_noise)
	pseudo_noise = pseudo_noise.type(dtype)

	def closure():
		global i, out_avg, psrn_noisy_last, last_net, net_input

		if reg_noise_std > 0:
			net_input = net_input_saved + (noise.normal_() * reg_noise_std)

		out = net(net_input)

		# Smoothing
		if out_avg is None:
			out_avg = out.detach()
		else:
			out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

		total_loss = mse(out, img_noisy_torch + pseudo_noise)
		total_loss.backward()
		error = torch.mean(out * pseudo_noise)
		rdPrint('%i\t%.7f' % (i, error), '{}/{}/pseudoErrorAtEpoch.log'.format(checkfolder, id_img))

		psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
		psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
		psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])
		rdPrint('%i\t%.7f' % (i, psrn_gt), '{}/{}/psrn_{}.log'.format(checkfolder, id_img, id_img))
		# Note that we do not have GT for the "snail" example
		# So 'PSRN_gt', 'PSNR_gt_sm' make no sense
		print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
		i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')

		out_np = torch_to_np(out)
		plot_image_grid([np.clip(out_np, 0, 1)], factor=6, nrow=1,
						it="{}/{}/id{}-{}".format(checkfolder, id_img, id_img, i))

		#if PLOT and i % show_every == 0:

		# Backtracking
		if i % show_every:
			if psrn_noisy - psrn_noisy_last < -5:
				print('Falling back to previous checkpoint.')

				for new_param, net_param in zip(last_net, net.parameters()):
					net_param.data.copy_(new_param.cuda())

				return total_loss * 0 , 0
			else:
				last_net = [x.detach().cpu() for x in net.parameters()]
				psrn_noisy_last = psrn_noisy

		i += 1

		return total_loss , 0


	p = get_params(OPT_OVER, net, net_input)
	optimize(OPTIMIZER, p, closure, LR, num_iter)

	out_np = torch_to_np(net(net_input))
	q = plot_image_grid([np.clip(out_np, 0, 1), img_np], nrow=2, factor=6,
						it="{}/{}/id{}-out".format(checkfolder, id_img, id_img));