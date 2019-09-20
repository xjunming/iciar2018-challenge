# -*- coding: utf-8 -*-
from __future__ import division
from utils import classes4_preview as preview_3
from utils import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
import os
from keras.models import load_model
import gc
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)
def weight_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    if not from_logits:
        output /= tf.reduce_sum(output, axis, True)
        _epsilon = _to_tensor(1e-7, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output)*(1 + tf.cast(tf.greater(tf.argmax(output),tf.argmax(target)),tf.float32)), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

def imagenet_processing(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:,:,i] -= mean[i]
        image[:,:,i] /= std[i]
    return image

'''
预测svs大图，得到预测的结果矩阵
输入：
    model：用于预测的模型
    svs_file_path: svs图片的保存路径
    name：svs图片的命名
输出：
    out_img：保存四类别预测结果的矩阵
'''    
def get_out_img(model, svs_file_path,name): 
    step1 = 512
    livel = 2
    slide = opsl.OpenSlide(svs_file_path)
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 
    
    w_count = int(slide.level_dimensions[0][0]) // step1
    h_count = int(slide.level_dimensions[0][1]) // step1
    out_img = np.zeros([h_count,w_count])
    i = 0
    for x in tqdm(range(w_count)):
        for y in range(h_count):
            i = i + 1          
            x0 =  x * step1
            y0 = y * step1
            slide_region1 = np.array(slide.read_region((x0, y0), 0, (step1, step1)))
            slide_img1 = slide_region1[:,:,:3]
            rgb_s1 = (abs(slide_img1[:,:,0] -107) >= 93) & (abs(slide_img1[:,:,1] -107) >= 93) & (abs(slide_img1[:,:,2] -107) >= 93)
            if np.sum(rgb_s1)<=(step1 * step1 ) * 0.5:
                img1 = cv2.resize(slide_img1, (224,224), interpolation = cv2.INTER_AREA)
                img1 = img1.reshape(1,224,224,3)
                prob = model.predict(imagenet_processing(img1/255))
                preIndex = np.argmax(prob, axis= 1)
                out_img[y,x] = preIndex[0]
                print(x, y, preIndex, prob)

    slide.close()
    out_img = cv2.resize(out_img, (int(w_count * step1 /Ds[livel,0]), int(h_count * step1 /Ds[livel,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[livel,1]-out_img.shape[0]),0,int(Wh[livel,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    out_img  = np.uint8(out_img)
    return out_img

'''
model_path:用于预测的模型保存路径
data_dir：保存所有svs文件以及xml文件的路径
save_base_dir：保存生成结果图片的路径
map_name: 四分类结果矩阵图片的前缀名
colormap_title:热图的标题名字
'''
def predict_svs(model_path, data_dir, save_base_dir, map_name, colormap_title):
    # model = load_model(model_path)
    model = load_model(model_path, custom_objects={'weight_categorical_crossentropy':weight_categorical_crossentropy})
    data = os.listdir(data_dir)
    for file in tqdm(data):
        if file.split('.')[-1] == 'svs':
            name = file.split('.')[-2]
            xml_name = name + '.xml'        
            svs_file = os.path.sep.join([data_dir, file])
            xml_file = os.path.sep.join([data_dir, xml_name]) 
            
            out_img = get_out_img(model,svs_file,name)
            pre_img = preview_3.get_preview(svs_file, xml_file)
       
            colormap_dir = os.path.sep.join([save_base_dir, 'colormap'])
            colormap_dir2 = os.path.sep.join([save_base_dir, 'invasive_colormap'])
            if not os.path.exists(colormap_dir):os.makedirs(colormap_dir)

            title = name + colormap_title
            colormap.cancer_invasive_show(pre_img, out_img, title,  colormap_dir) # 4 class predict

            del out_img, pre_img
            gc.collect()
# model_path = '/cptjack/totem/yatong/model/cnn_model/resnet50_newhsv_0813/resnet50.h5'
model_path = './result/190920/resnet50.h5'
save_base_dir = '/cptjack/totem/xjunming/ICIAR/iciar2018-challenge/result/190919/predicted'   
data_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/WSI/A_'
map_name = 'resnet50_map.png'
colormap_title = '_resnet50_colormap'

predict_svs(model_path, data_dir, save_base_dir, map_name, colormap_title)

