# -*- coding: utf-8 -*-
# @Time	: 2019.9.18
# @Author  : Xie Junming
# @Licence : bio-totem

from PIL import Image
import numpy as np
from skimage import io
from imutils import paths
import os
from tqdm import tqdm
import re
import cv2
import concurrent.futures
import time

step = 512
patch_size = 512
scale_range = 0.6
stain_channels = ['h', 's', 'v']
aug_num = 2
read_base_dir = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/'

save_base_dir = "./data/dataset_step_%(step)s_patch_%(patch)s_scale_%(scale)s" \
                "_channels_%(channels)s_aug_%(aug)s" % {'step': step, 'patch': patch_size, 'scale': scale_range,
                                                            'channels': ''.join(stain_channels), 'aug': aug_num}

class data_preprocessing:
    def __init__(self,
                 step=512,
                 patch_size=512,
                 scale_range=0,
                 stain_channels=['h', 's', 'v'],
                 aug_num=2):
        """
        对原数据集进行截图、颜色增强，并保存为自己的数据集。
        由于我开启了多进程，因此调用时需要在if __name__ == '__main__':里面运行，
        若果你不想开启多线程，你可以在line 144~151进行修改，具体可以参考./demo/data_processing_demo.py
        # Arguments
            img_list: 图像的文件路径
            step: 截图移动的步长，默认512
            patch_size: 截图保存的像素大小，默认512
            scale_range: 染色的方差，原图就会乘以一个[1 - scale_range, 1 + scale_range]内的一个随机数作为颜色增强，默认不做颜色增强
            stain_channels: 染色的通道，默认h, s, v 通道
            aug_num: 颜色增强的次数
        """

        self.step = step
        self.patch_size = patch_size
        self.scale_range = scale_range
        self.stain_channels = stain_channels
        self.aug_num = aug_num

    def get_scale(self):
        while 1:
            scale = np.random.uniform(low=1.08-self.scale_range, high=1.08+self.scale_range)
            if abs(scale-1)>0.012:
                break
        return scale

    def hsv_aug(self, img):
        """
        对图片进行颜色增强。
        :param img: Img矩阵，注意，输入的Img矩阵的三个通道默认为RGB，即和opencv默认读取的通道一致
        :return: 染色增强后的img矩阵
        """

        if self.scale_range == 0:
            return 0
        else:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if 'h'in self.stain_channels:
                scale = self.get_scale()
                hsv_img[:, :, 0] = hsv_img[:, :, 0] * scale
            elif 's' in self.stain_channels:
                scale = self.get_scale()
                hsv_img[:, :, 1] = hsv_img[:, :, 1] * scale
            elif 'v' in self.stain_channels:
                scale = self.get_scale()
                hsv_img[:, :, 2] = hsv_img[:, :, 2] * scale
                hsv_img[:, :, 2] = hsv_img[:, :, 2] * (hsv_img[:, :, 2] < 255) + (hsv_img[:, :, 2] >= 255)*255

            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
            return img

    def get_patch(self, file, result_dir, test_data=False):
        """
        对图片进行切割，获取patch，并对其进行保存操作。file为大图的文件名称，图片保存的方式举个例子，对test.png进行截图，
        则保存为test_x_y_.png，染色增强的图片保存为test_x_y_0.png，其中，x,y为截图在原图中所对应的坐标。
        :param file: 大图的文件名称
        :param result_dir: 截图后保存的文件夹名称
        :param test_data: 是否为测试集数据，若为True，则不进行颜色增强
        :return: 最后保存截图及其颜色增强的图片
        """
        f = re.split(r'/|\\', file)[-1]
        f_name = f.split('.')[-2]
        img = Image.open(file)
        img = np.asarray(img)
        h_count = img.shape[0] // self.step
        w_count = img.shape[1] // self.step
        for y in range(h_count):
            for x in range(w_count):
                x0 = x * self.step
                x1 = x0 + self.patch_size
                y0 = y * self.step
                y1 = y0 + self.patch_size
                patch = img[y0:y1, x0:x1, :]
                rgb_s = (abs(patch[:, :, 0] - 107) >= 93) & (abs(patch[:, :, 1] - 107) >= 93) & (
                        abs(patch[:, :, 2] - 107) >= 93)
                if np.sum(rgb_s) >= (self.patch_size * self.patch_size) * 0.6:
                    continue
                if patch.shape != (self.patch_size, self.patch_size, 3):
                    continue
                elif test_data:
                    io.imsave(result_dir + '/' + f_name + '_' + str(x) + '_' + str(y) + '_.png', patch)
                    continue
                else:
                    io.imsave(result_dir + '/' + f_name + '_' + str(x) + '_' + str(y) + '_.png', patch)

                for i in range(self.aug_num):
                    save_path = result_dir + '/' + f_name + '_' + str(x) + '_' + str(y) + '_' + str(i) + '.png'
                    patch_save = self.hsv_aug(patch)
                    if np.sum(patch_save):
                        io.imsave(save_path, patch_save)

    def cut_data(self, img_list, save_dir, test_data=False):
        """
        主要函数，遍历img_list里所有的大图文件，对其进行切割操作，同时进行染色变换。
        由于涉及到很多I/O操作(图片的读取和保存)，所以我在这里用了多进程操作，以提高图片处理速度。
        :param img_list: 大图的文件名组成的list
        :param save_dir: 图片增强后保存的文件夹路径
        :param test_data: 默认为False, 如果为True, 则只进行切图，不进行染色，以缩短验证时间
        :return: None
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print('\nsaving in %s'%save_dir)
        start_time = time.asctime(time.localtime(time.time()))

        # 开启多进程
        with concurrent.futures.ProcessPoolExecutor(30) as executor:
             for img in img_list:
                 executor.submit(self.get_patch, img, save_dir, test_data)

        # 不开启多进程
        # for i in tqdm(img_list):
        #     self.get_patch(i, save_dir, test_data=test_data)

        print('\nstart at %s'%start_time)
        print('\nend at %s'%time.asctime(time.localtime(time.time())))


if __name__=='__main__':
    """
    这里主要涉及到文件读取和文件夹创建的操作
    """
    def read_file(filename='../ICIAR_visualization/classify/class_0'):
        with open(filename) as f:
            train_paths = []
            line = f.readline().rstrip('\n')
            line = re.split(r'/|\\', line)[-1]
            train_paths.append(line)
            while line:
                line = f.readline().rstrip('\n')
                line = re.split(r'/|\\', line)[-1]
                train_paths.append(line)
            return train_paths

    train_dir = os.path.sep.join([save_base_dir, 'train'])
    test_dir = os.path.sep.join([save_base_dir, 'test'])
    CLASSES = ["Normal", "Benign", "Insitu", "Invasive", ]
    read_dir, train_class_dir, test_class_dir = [], [], []
    for v in CLASSES:
        read_dir.append(read_base_dir + v)
        train_class_dir.append(os.path.sep.join([train_dir, v]))
        test_class_dir.append(os.path.sep.join([test_dir, v]))
        if not os.path.exists(os.path.sep.join([train_dir, v])):
            os.makedirs(os.path.sep.join([train_dir, v]))
        if not os.path.exists(os.path.sep.join([test_dir, v])):
            os.makedirs(os.path.sep.join([test_dir, v]))

    train_paths = {}
    for i, v in enumerate(read_dir):
        train_paths[i] = list(paths.list_images(v))

    # load test data
    files = [
        "/cptjack/totem/xjunming/ICIAR/ICIAR_visualization/classify/test"
    ]
    val_type = []
    for f in files:
        val_type.extend(read_file(f))
    val_type = [i for i in val_type if i != '']
    test_paths = {}
    for t in train_paths:
        test_paths[t] = []
        for i, v in enumerate(train_paths[t]):
            if re.split(r'/|\\', v)[-1] in val_type:
                test_paths[t].append(v)
                train_paths[t].remove(v)
            else:
                pass

    print("\n start cutting pics")
    p = data_preprocessing(
                 step=step,
                 patch_size=patch_size,
                 scale_range=scale_range,
                 stain_channels=stain_channels,
                 aug_num=aug_num)

    for i in tqdm(train_paths):
        p.cut_data(train_paths[i], save_dir=train_class_dir[i])
    for i in tqdm(test_paths):
        p.cut_data(test_paths[i], save_dir=test_class_dir[i], test_data=True)
