# Xi Peng, Feb 2017
# Yu Tian, Apr 2017
import os, sys
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

dd = pdb.set_trace

views = ['200', '190', '041', '050', '051', '140', '130', '080', '090']

pi = 3.1416 # 180 degree
d_60 = pi / 3
d_15 = pi / 12
d_range = pi / 36 # 5 degree

d_45 = d_60 - d_15
d_30 = d_45 - d_15

def read_img(img_path):
    # img_path: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128,128), Image.ANTIALIAS)
    return img

def get_multiPIE_img(img_path):

    # img_path: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
    tmp = random.randint(0, 8)
    view2 = tmp

    view = views[tmp]

    token = img_path.split('/')
    name = token[-1]
        
    token = name.split('_')
    ID = token[0]
    status = token[2]
    bright = token[4]
        
    img2_path = '/home/yt219/data/multi_PIE_crop_128/' + ID + '/' + ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'
    img2 = read_img( img2_path )
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    return view2, img2

def get_300w_LP_img(img_path):
    # img_path = '/home/yt219/data/crop_0822/AFW_resize/AFW_1051618982_1_0_128.jpg'
    # txt_path: /home/yt219/data/300w_LP_size_128/AFW_resize/AFW_1051618982_1_0_128_pose_shape_expression_128.txt 
    right = img_path.find('_128.jpg')
    for i in range(right-1, 0, -1):
        if img_path[i] == '_':
            left = i
            break
    
    view2 = -1
    while(view2 < 0):
        tmp = random.randint(0, 17)
        new_txt = img_path[:left+1] + str(tmp) + '_128_pose_shape_expression_128.txt'
        new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")
        
        if os.path.isfile(new_txt):
            param = np.loadtxt(new_txt)
            yaw = param[1]
            if yaw < -d_60 or yaw > d_60:
                view2 = -1
            elif yaw >= -d_60 and yaw < -d_60+d_range:
                view2 = 0
            elif yaw >= -d_45-d_range and yaw < -d_45+d_range:
                view2 = 1
            elif yaw >= -d_30-d_range and yaw < -d_30+d_range:
                view2 = 2
            elif yaw >= -d_15-d_range and yaw < -d_15+d_range:
                view2 = 3
            elif yaw >= -d_range and yaw < d_range:
                view2 = 4
            elif yaw >= d_15-d_range and yaw < d_15+d_range:
                view2 = 5
            elif yaw >= d_30-d_range and yaw < d_30+d_range:
                view2 = 6
            elif yaw >= d_45-d_range and yaw < d_45+d_range:
                view2 = 7
            elif yaw >= d_60-d_range and yaw <= d_60:
                view2 = 8
    
    new_img = img_path[:left+1] + str(tmp) + '_128.jpg'
    img2 = read_img( new_img )
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    
    return view2, img2

class ImageList(data.Dataset):
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[128,128] ):
        img_list = [line.rstrip('\n') for line in open(list_file)]
        print('total %d images' % len(img_list))

        self.img_list = img_list
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])

    def __getitem__(self, index):
        # img_name: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
        img1_path = self.img_list[index]
        
        img1 = read_img( img1_path )

        if self.transform_img is not None:
            img1 = self.transform_img(img1) # [0,1], c x h x w

        return img1

    def __len__(self):
        return len(self.img_list)
