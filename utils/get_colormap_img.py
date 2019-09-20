# -*- coding: utf-8 -*-


import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

def create_colormap(svs_im, matrix_0, title, output_dir):
    plt_size = (svs_im.size[0] // 100, svs_im.size[1] //100)
    flg, ax = plt.subplots(figsize = plt_size, dpi =100)
    matrix = matrix_0.copy()
    matrix = cv2.resize(matrix, svs_im.size, interpolation = cv2.INTER_AREA)
    cax = ax.imshow(matrix, cmap = plt.cm.jet, alpha = 0.45)  
    svs_im_npy = np.array(svs_im.convert('RGBA'))  
    svs_im_npy[:,:][matrix[:,:] > 0] = 0  
    ax.imshow(svs_im_npy) 
    max_matrix_value = matrix.max() 
    plt.colorbar(cax, ticks = np.linspace(0, max_matrix_value, 25, endpoint = True)) 
    ax.set_title(title, fontsize = 20)
    plt.axis('off')
    
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title ))
    plt.close('all')

def index_color(index):
    color = [0, 0, 0, 255]
    area = int(index / 255)
    move = index % 255
    baseloc = int(area / 2)
    exloc = int((area / 2 + area % 2)%3)
    ready = int(2- ((1 + area) % 3))
    sign = 1 - area % 2 *2
    color[baseloc] = 255
    color[exloc] = 255
    color[ready] += sign*move
    return color

def cancer_invasive_show(svs_img, cancer_invasive_matrix, title, output_dir):
    labels = ['Benign', 'InSitu', 'Invasive']
    #color_irc = int(1020/len(labels))
    #colors = np.array([index_color(i*color_irc) for i in range(len(labels))])
    colors = np.array([[0, 0, 225, 150], [0, 225, 0, 150], [225, 0, 0, 150]])  #blue, green, red
    svs_img_bg = svs_img.resize((cancer_invasive_matrix.shape[1], cancer_invasive_matrix.shape[0]))
    patches = [mpatches.Patch(color=colors[i] / 255, label="{:s}".format(labels[i])) for i in range(len(labels))]
    cancer_invasive_image = np.full((cancer_invasive_matrix.shape[0], cancer_invasive_matrix.shape[1], 4), 0,
                                    dtype=np.uint8)
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 1] = colors[0]  # Benign color
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 2] = colors[1]  # InSitu color
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 3] = colors[2]  # Invasive color
    plt_size = (cancer_invasive_matrix.shape[1] // 200, cancer_invasive_matrix.shape[0] // 200)
    fig, ax = plt.subplots(figsize=plt_size, dpi=200)
    ax.imshow(svs_img_bg)
    ax.imshow(cancer_invasive_image)
    ax.legend(handles=patches)
    ax.set_title(title, fontsize=20)
    plt.axis('off')
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))
    plt.close('all')
