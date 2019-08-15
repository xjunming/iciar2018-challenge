# -*- coding: utf-8 -*-
from keras.models import load_model
#from utils import metrics
from utils1 import generators


#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adagrad, SGD
from keras.utils import to_categorical
from keras.applications import resnet50
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import regularizers
from keras.utils.vis_utils import plot_model
from keras.models import Sequential

from keras.layers.convolutional import  MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Flatten, Dropout 
import random
#from keras.models import Model, Input
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint
import time
import tensorflow as tf
#from generators import DataGenerator




os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/resnet50(224).h5'
#file_path = '/cptjack/sys_software_bak/tensorflow_keras_models/models/InceptionResnetV2(224).h5'
#
base_model = load_model(file_path)
base_model.summary()
#plot_model(base_model, to_file='model1.png', show_shapes=True)
model = Sequential()
model.add(base_model)
#top_model.add(BatchNormalization())
#top_model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

#top_model.add(Dense(16,activation='relu',kernel_initializer='he_normal',
#                    kernel_regularizer=regularizers.l2(0.01)))
#top_model.add(Dense(16,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(32,activation='relu',kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.summary()


for layer in model.layers:
    layer.trainable = True
    
#model_path = '/cptjack/totem/yatong/4_classes/inceResV2_0806_2/InceptionResnetV2(224).h5'
#model = load_model(model_path)
##
#model.summary()


#for layer in model.layers:
#    layer.trainable = True


num_epochs = 30
init_lr = 1e-2
bs = 32

#train_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset_512/train'
#val_dir = '/cptjack/totem/yatong/all_data/balance_normalized_dataset_512/validation'
#train_dir = '/cptjack/totem/yatong/all_data/bach_augment_data_512/train'
#val_dir = '/cptjack/totem/yatong/all_data/bach_augment_data_512/validation'
#train_dir = '/cptjack/totem/yatong/all_data/mil_new_512/train'
val_dir = '/cptjack/totem/yatong/all_data/val'
train_dir = '/cptjack/totem/yatong/all_data/new_hsv_augment_data_512/train'
#train_dir ='/cptjack/totem/yatong/all_data/mil_data_512/train'
#val_dir = '/cptjack/totem/yatong/all_data/val'
#train_path = config2.train_dir
opt = SGD(lr=init_lr, decay=init_lr/num_epochs, 
                     momentum=0.9, nesterov=True)

#opt = Adagrad(lr=init_lr, decay=init_lr / num_epochs)
model.compile(loss = "weight_categorical_crossentropy", optimizer = opt,
                  metrics = ["accuracy"])
#top_model.compile(loss="categorical_crossentropy", optimizer=opt,
#              metrics=["accuracy"]
#              )

trainPaths = list(paths.list_images(train_dir))
random.seed(40)
random.shuffle(trainPaths)
#trainPaths = list(paths.list_images(new_train.creat_train_dir))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(val_dir)))
#totalTest = len(list(paths.list_images(test_dir)))

#trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
#print(trainLabels[:3])
#trainLabels = to_categorical(trainLabels)
#print(trainLabels[:3])
#classTotals = trainLabels.sum(axis = 0)
#classWeight = classTotals.max()/classTotals

trainAug = generators.DataGenerator(rescale=1/255.0)
                           #         rotation_range=20,
#                              width_shift_range=0.1,
#                              height_shift_range=0.1,
#                              shear_range=0.05,
#                              horizontal_flip=True,
#                              vertical_flip=True,
#                              zoom_range = 0.2,
#                              stain_transformation = True,
#                              fill_mode="nearest")

valAug = generators.DataGenerator(rescale=1/255.0)

trainGen = trainAug.flow_from_directory(train_dir,
                                        class_mode="categorical",
                                        target_size=(224,224),
                                        color_mode="rgb",
                                        shuffle=True,
                                        batch_size=bs)


valGen = valAug.flow_from_directory(val_dir,
                                        class_mode="categorical",
                                        target_size=(224,224),
                                        color_mode="rgb",
                                        shuffle=True,
                                        batch_size=bs)

#testGen = valAug.flow_from_directory(config.test_dir,
#                                        class_mode="categorical",
#                                        target_size=(224,224),
#                                        color_mode="rgb",
#                                        shuffle=False,
#                                        batch_size=bs)


class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_acc',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model,'./resnet50_newhsv_0813/'+ filepath + '.h5') 
    file_dir = './resnet50_newhsv_0813/log/'+str(filepath) + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./resnet50_newhsv_0813/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_'+str(filepath) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

#file = 'InceptionResnetV2(224)'
file = 'resnet50'
callbacks_s = get_callbacks(file,model,patience=10)
#early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='min')
H = model.fit_generator(trainGen,
                        steps_per_epoch=totalTrain // bs,
                        validation_data=valGen,
                        validation_steps=totalVal // bs,
#                        class_weight=classWeight,
                        epochs=num_epochs,
                        callbacks=callbacks_s,
                        verbose=1)

#print("[info] evaluating network..")
#testGen.reset()
#predIdxs = top_model.predict_generator(testGen,
#                                   steps=(totalTest // bs) + 1)
#
#predIdxs = np.argmax(predIdxs, axis=1)
#
#print(classification_report(testGen.classes, predIdxs, 
#                            target_names=testGen.class_indices.keys()))
#
#cm = confusion_matrix(testGen.classes, predIdxs)
#total = sum(sum(cm))
#acc = (cm[0,0] + cm[1,1]) / total
#sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
#specificity = cm[1,1] / (cm[1,0] + cm[1,1])
#
#print(cm)
#print("acc:{:.4f}".format(acc))
#print("sensitivity:{:.4f}".format(sensitivity))
#print("specificity:{:.4f}".format(specificity))






