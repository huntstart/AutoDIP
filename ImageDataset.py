'''
copy from imageDataset3, for classification task.
'''
import torch
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as tf
import PIL.Image
from utils.common_utils import *
# import torch.tensor

class ImageDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self,listFile='../data/coco-train2017.list',StartAt=0,Nub=1):
        # TODO
        # 1. Initialize file path or list of file names.
        self.listFile = listFile
        self.sa = StartAt
        self.Nub = Nub

        print(self.listFile)

        self.name_list = []
        self.id_list = []
       
        file = open(self.listFile,'r',encoding='utf-8')  
        line = file.readline()
        line = line.strip('\n')
        self.root = line
        line = file.readline()
        while line:
            line = line.strip('\n')
            if line[0] != '#':
                temp_id = int(line[0:line.find(': ')])
                line = line[line.find(': ')+2:]
                fh = line
                
                if fh != None:
                    self.id_list.append(temp_id)
                    self.name_list.append(fh)
            else:
                print('pass==> {}'.format(line))
#                     self.posi_list.append('pass')
            line = file.readline()

        file.close()
            
        self.set_len = self.name_list.__len__()
        if self.sa+self.Nub>self.set_len:
            self.set_len = self.set_len-self.sa
        else:
            self.set_len = self.Nub

        print(self.set_len)

        
#         self.initPerm()
        
#     def initPerm(self):
#         self.set_perm = torch.randperm(self.set_len)

#     def getIndex(self,index):
#         index = self.set_perm[index%self.set_len]
#         return index

    def loadImage(self,path):
        img = PIL.Image.open(path)
        img = crop_image(img, d=32)  # @UndefinedVariable

        return img

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        # shuffle in this part is not necessary.

#         index,Target = self.getIndex(index)
        index = index+self.sa
        
        img = self.loadImage(self.root+'/'+self.name_list[index])
        img_id = self.id_list[index]
        img_name  = self.name_list[index]
        
        return img,img_id,img_name
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.set_len
