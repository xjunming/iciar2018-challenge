# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from skimage import io
#import cv2
import os
from imutils import paths


#save_base_dir = '/cptjack/totem/yatong/new_data/dataset'
save_base_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset_512'
train_dir = os.path.sep.join([save_base_dir, 'train'])
val_dir = os.path.sep.join([save_base_dir, 'validation'])

train_class0_dir = os.path.sep.join([train_dir, '0'])
train_class1_dir = os.path.sep.join([train_dir, '1'])
train_class2_dir = os.path.sep.join([train_dir, '2'])
train_class3_dir = os.path.sep.join([train_dir, '3'])

val_class0_dir = os.path.sep.join([val_dir, '0'])
val_class1_dir = os.path.sep.join([val_dir, '1'])
val_class2_dir = os.path.sep.join([val_dir, '2'])
val_class3_dir = os.path.sep.join([val_dir, '3'])

if not os.path.exists(train_class0_dir):os.makedirs(train_class0_dir)
if not os.path.exists(train_class1_dir):os.makedirs(train_class1_dir)
if not os.path.exists(train_class2_dir):os.makedirs(train_class2_dir)
if not os.path.exists(train_class3_dir):os.makedirs(train_class3_dir)

if not os.path.exists(val_class0_dir):os.makedirs(val_class0_dir)
if not os.path.exists(val_class1_dir):os.makedirs(val_class1_dir)
if not os.path.exists(val_class2_dir):os.makedirs(val_class2_dir)
if not os.path.exists(val_class3_dir):os.makedirs(val_class3_dir)

#train_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset/train'
train_dir = '/cptjack/totem/yatong/all_data/bach_data_color_norm/validation'
train_0 = os.path.sep.join([train_dir, '0'])
train_1 = os.path.sep.join([train_dir, '1'])
train_2 = os.path.sep.join([train_dir, '2'])
train_3 = os.path.sep.join([train_dir, '3'])
#if not os.path.exists(train_1):os.makedirs(train_1)
#if not os.path.exists(train_0):os.makedirs(train_0)
train_paths_0 = list(paths.list_images(train_0))
train_paths_1 = list(paths.list_images(train_1))
train_paths_2 = list(paths.list_images(train_2))
train_paths_3 = list(paths.list_images(train_3))

#val_dir = '/cptjack/totem/yatong/bach_data_color_norm/validation'
#val_1 = os.path.sep.join([val_dir, '1'])
#val_0 = os.path.sep.join([val_dir, '0'])
#if not os.path.exists(val_1):os.makedirs(val_1)
#if not os.path.exists(val_0):os.makedirs(val_0)
#val_paths_1 = list(paths.list_images(val_1))
#val_paths_0 = list(paths.list_images(val_0))
##result = '/cptjack/totem/yatong/new_data/result'

def get_patch(file, result_dir):
#    l = file.split('/')[-2]
#    print(l)
    f = file.split('/')[-1]
    f_name = f.split('.')[-2]
    img = Image.open(file)
    img = np.asarray(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
    i = 0 
    print(f_name,h_count, w_count)
    
    for y in range(h_count):
        for x in range(0,w_count):
            x0 =  x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
#            print(x0,x1,y0,y1)
            patch = img[y0:y1, x0:x1]
            rgb_s = (abs(patch[:,:,0] -107) >= 93) & (abs(patch[:,:,1] -107) >= 93) & (abs(patch[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<=(step * step ) * 0.6:
                i = i + 1
#                io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
                io.imsave(result_dir + '/' + f_name + '_'+str(i) +'.png', patch)
    return


  
    
def get_dataset(paths,save_dir):
    for file in paths:
        get_patch(file,save_dir)
        
#get_dataset(build_dataset2.val_class1_paths, val_class1_dir)
#get_dataset(build_dataset2.val_class0_paths, val_class0_dir)
#get_dataset(build_dataset3.train_class1_paths, train_class1_dir)
#get_dataset(build_dataset3.train_class0_paths, train_class0_dir)   

#get_dataset(train_paths_0, train_class0_dir)
#get_dataset(train_paths_1, train_class1_dir)
#get_dataset(train_paths_2, train_class2_dir)
#get_dataset(train_paths_3, train_class3_dir)

get_dataset(train_paths_0, val_class0_dir)
get_dataset(train_paths_1, val_class1_dir)
get_dataset(train_paths_2, val_class2_dir)
get_dataset(train_paths_3, val_class3_dir)

#file = '/cptjack/totem/yatong/all_data/balance_normalized_dataset/train/0/Normal 40x_Y243_5.tif'
#get_patch(file, train_class1_dir)
#get_dataset(val_paths_1, val_class1_dir)
#get_dataset(val_paths_0, val_class0_dir)
##def get_patch(file, result_dir):
#    l = file.split('/')[-2]
#    print(l) 
#    f = file.split('/')[-1]
#    f_name = f.split('.')[-2]
#    img = Image.open(file)
#    img = np.asarray(img)
#    step = 512
#    h_count = img.shape[0] // step
#    w_count = img.shape[1] // step
#    i = 0 
#    for y in range(h_count):
#        for x in range(0,w_count-1):
#            x0 = 256 + x * step
#            x1 = x0 + step
#            y0 = y * step
#            y1 = y0 + step
##            print(x0,x1,y0,y1)
#            patch = img[y0:y1, x0:x1]
#            i = i + 1
#            io.imsave(result_dir + '/' + l +'_'+ f_name + '_'+str(i) +'.png', patch)
#    return
#
#
#  
#    
#def get_dataset(paths,save_dir):
#    for file in paths:
#        get_patch(file,save_dir)
#        
#get_dataset(build_dataset.val_class1_paths, val_class1_dir)
#get_dataset(build_dataset.val_class0_paths, val_class0_dir)
#get_dataset(build_dataset.train_class1_paths, train_class1_dir)
#get_dataset(build_dataset.train_class0_paths, train_class0_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    