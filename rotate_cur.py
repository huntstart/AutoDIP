#18
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import math
import  torch.utils.data.dataset as DataSet
import torch.utils.data.dataloader as DataLoader
#p = np.loadtxt('./../psrn/psrn.log')
import sys

# 打印修改时间



def averagenum(num):
	nsum = 0
	for i in range(len(num)):
		nsum += num[i]
	return nsum / len(num)


def rdPrint(anyThing, path='./'):
	orig_stdout = sys.stdout
	f = open(path, 'a')
	sys.stdout = f

	print(anyThing)

	sys.stdout = orig_stdout
	f.close()


def curvature(f, x0):
	nuL = len(f) - 1
	temp = 0
	for i in range(nuL - 2 + 1):
		temp2 = 1
		for j in range(nuL - i - 2 + 1, nuL - i + 1):
			temp2 = temp2 * j  # ！！！！！！！要验证一下是否正确
		temp += f[i] * np.power(x0, (nuL - 2 - i)) * temp2
	# print(temp)
	return temp

def rotate(x, y):
	xnew = x[:].copy()
	ynew = y[:].copy()
	for i in range(len(x)):
		xnew[i] = (x[i] * math.cos(dg) + y[i] * math.sin(dg) + a)
		ynew[i] = (-x[i] * math.sin(dg) + y[i] * math.cos(dg) + b)
	return xnew, ynew

halfwindow = 400
avg = 40  # 十分之1半窗口
level = 2
amp = 800 / 0.0025
notCount = 600
checkfolder_root = '../denoise-set1'#use the file name
print (checkfolder_root)
filename = '{}/best-eopch.txt'.format(checkfolder_root)
file2 = open(filename, 'w', encoding='utf-8')
file2.writelines('{}\t{}\t{}\t{}\t{}\t{}'.format('imgID','maxCurErr','maxCurPsr','maxEpc','PsrRate','EndRate')+ '\n')

file = open('../data/denoise.list','r',encoding='utf-8')
line = file.readline()  # 读list第一行
line = line.strip('\n')  # 删除首尾换行
line = file.readline()  # line为读到的list第二行
while line:
	line = line.strip('\n')
	id_img = int(line[0:line.find(': ')])
	checkfolder = '{}/{}'.format(checkfolder_root,id_img)
	error = np.loadtxt('{}/pseudoErrorAtEpoch.log'.format(checkfolder))
	psrn = np.loadtxt('{}/psrn_{}.log'.format(checkfolder,id_img))
	#	for super-resolution
	# psrn = np.loadtxt('{}/psrn_HR.log'.format(checkfolder, id_img))
	
#if there is a breakpoint,stop at it
	for i in range(notCount,error.shape[0]-10):
		if error[i,1]-error[i+10,1]>0.000015:
			error = error[:i]
			psrn = psrn[:i]
			break






	cur = error[:,1].copy()
	er = error[:,1].copy()
	ps = psrn[:,1].copy()
	cur.fill(0)

	for i in range(halfwindow + 0, error.shape[0] - halfwindow, 1):
		tempX = error[i - halfwindow:i + halfwindow, 0].copy()
		tempY = error[i - halfwindow:i + halfwindow, 1].copy() * amp
		tempX = tempX - i

		# 求需要的首中尾三点
		x1 = i - halfwindow + avg / 2
		YF = error[i - halfwindow:i - halfwindow + avg, 1].copy() * amp  # Y_Frist
		y1 = averagenum(YF)

		x2 = 0
		YM = error[i - avg // 2:i + avg // 2, 1].copy() * amp  # Y_Mid
		y2 = averagenum(YM)

		x3 = i + halfwindow - avg / 2
		YL = error[i + halfwindow - avg:i + halfwindow, 1].copy() * amp  # Y_LAST
		y3 = averagenum(YL)

		#	dg = math.degrees(math.atan2(y1-y3,x1-x3))
		dg = math.atan2(y3 - y1, x3 - x1)  # degrees
		m1 = (x1 - x2) * math.cos(dg) + (y1 - y2) * math.sin(dg)
		m2 = (x3 - x2) * math.sin(dg) + (y2 - y3) * math.cos(dg)
		m3 = (x3 - x2) * math.cos(dg) + (y3 - y2) * math.sin(dg)
		a = -(x2 * math.cos(dg) + y2 * math.sin(dg))
		b = x3 * math.sin(dg) - y3 * math.cos(dg)
		tempX1, tempY1 = rotate(tempX, tempY)

		f = np.polyfit(tempX1, tempY1, level)
		cur[i] = curvature(f, 0)
		rdPrint('%i\t%.7f' % (i, curvature(f, 0)), '{}/cur.log'.format(checkfolder))

	

	print ('notCount:',notCount)
	max_index = np.argmax(np.array(cur[notCount:]))+notCount
	print(max_index)
	eMax = np.max(error[notCount:,1])
	cMax = np.max(cur[notCount:])
	pMax = np.max(psrn[notCount:,1])
	plt.figure(1)
	plt.clf()
	#plt.annotate(show_max,xytext=(1,2),xy=(max_index,cur[max_index]))
# 	file2 = open(filename, 'a', encoding='utf-8')
	#file2.writelines('{}\t{:.4f}\t{:.4f}\t{}\t{:.4f}\t{:.4f}'.format(id_img,er[max_index],ps[max_index],max_index,ps[max_index]/np.max(ps),ps[-1]/np.max(ps))+ '\n')
	file2.writelines('{}\t{:.4f}\t{:.4f}\t{}\t{:.4f}\t{:.4f}'.format(id_img,er[max_index],ps[max_index],max_index,ps[max_index]/pMax,ps[-1]/pMax)+ '\n')
	testA=cur*eMax/cMax
	testA[testA<0]=0
	testP = psrn[notCount:,1].copy()*eMax/pMax
	plt.plot(max_index,eMax,'ks')
	show_max='['+str(max_index)+' , '+str(cur[max_index])+']'
	plt.annotate(show_max,xy=(max_index,testA[max_index]))

	plt.plot(error[notCount:,0],testA[notCount:],'b')

	plt.plot(error[notCount:,0],error[notCount:,1],'r')
	plt.plot(error[notCount:,0],testP[:],'y')

	plt.savefig('{}/2_400.png'.format(checkfolder))
	#plt.close()
	line = file.readline()

file.close()
file2.close()
