from PIL import Image
import numpy as np
from skimage import io
import os
import sys
from imutils import paths
from tqdm import tqdm

def get_path(argv = sys.argv):
    if len(argv)>1:
        # Linux
        if 'L' in sys.argv[1]:
            path = '/cptjack/totem/Colon Pathology/openslide_test/ICIAR2018_BACH_Challenge/Train/Photos/'
            print(' %s running in \033[1;35m Linux \033[0m!'%argv[0])
    else:
        # Windows
        path = '../data'
        print(' %s running in \033[1;35m Windows \033[0m!'%argv[0])
    return path

def get_patch(file, result_dir):
    f = file.split('\\')[-1]
    f_name = f.split('.')[-2]
    img = Image.open(file)
    img = np.asarray(img)
    step = 512
    h_count = img.shape[0] // step
    w_count = img.shape[1] // step
    i = 0
    # print(f_name, h_count, w_count)
    for y in range(h_count):
        for x in range(0, w_count):
            x0 = x * step
            x1 = x0 + step
            y0 = y * step
            y1 = y0 + step
            patch = img[y0:y1, x0:x1]
            rgb_s = (abs(patch[:, :, 0] - 107) >= 93) & (abs(patch[:, :, 1] - 107) >= 93) & (
                        abs(patch[:, :, 2] - 107) >= 93)
            if np.sum(rgb_s) <= (step * step) * 0.6:
                i = i + 1
                io.imsave(result_dir + '/' + f_name + '_' + str(i) + '.png', patch)
    return

def get_dataset(paths, save_dir):
    for file in tqdm(paths):
        get_patch(file, save_dir)

save_base_dir = '../balance_normalized_dataset_512'
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



def read_file(filename='../ICIAR_visualization/classify/class_0'):
    with open(filename) as f:
        train_paths = []
        line = f.readline().rstrip('\n').split('/')[-1]
        train_paths.append(line)
        while line:
            line = f.readline().rstrip('\n').split('/')[-1]
            train_paths.append(line)
        return train_paths

files = [
    '../ICIAR_visualization/classify/class_0','../ICIAR_visualization/classify/class_1',
    '../ICIAR_visualization/classify/class_4'
]
val_type = []
for f in files:
    val_type.extend(read_file(f))
train_dir = get_path()
try:
    train_0 = os.path.sep.join([train_dir, 'Normal'])
    train_1 = os.path.sep.join([train_dir, 'Benign'])
    train_2 = os.path.sep.join([train_dir, 'Insitu'])
    train_3 = os.path.sep.join([train_dir, 'Invasive'])
except:
    train_0 = train_dir + '/Normal/'
    train_1 = train_dir + '/Benign/'
    train_2 = train_dir + '/Insitu/'
    train_3 = train_dir + '/Invasive/'

train_paths_0 = list(paths.list_images(train_0))
train_paths_1 = list(paths.list_images(train_1))
train_paths_2 = list(paths.list_images(train_2))
train_paths_3 = list(paths.list_images(train_3))

train_paths_0 = [i for i in train_paths_0 if i.split('\\')[-1] not in val_type]
test_paths_0 = [i for i in train_paths_0 if i.split('\\')[-1] in val_type]
train_paths_1 = [i for i in train_paths_1 if i.split('\\')[-1] not in val_type]
test_paths_1 = [i for i in train_paths_1 if i.split('\\')[-1] in val_type]
train_paths_2 = [i for i in train_paths_2 if i.split('\\')[-1] not in val_type]
test_paths_2 = [i for i in train_paths_2 if i.split('\\')[-1] in val_type]
train_paths_3 = [i for i in train_paths_3 if i.split('\\')[-1] not in val_type]
test_paths_3 = [i for i in train_paths_3 if i.split('\\')[-1] in val_type]

# train data
get_dataset(train_paths_0, train_class0_dir)
get_dataset(train_paths_1, train_class1_dir)
get_dataset(train_paths_2, train_class2_dir)
get_dataset(train_paths_3, train_class3_dir)

# test data
get_dataset(test_paths_0, val_class0_dir)
get_dataset(test_paths_1, val_class1_dir)
get_dataset(test_paths_2, val_class2_dir)
get_dataset(test_paths_3, val_class3_dir)
