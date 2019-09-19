from keras.models import load_model
from utils import generators
from keras.callbacks import CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
import os
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
import time
import datetime
from utils.resnet import ResNet18
from imutils import paths
import random
import tensorflow as tf

num_epochs = 50
init_lr = 1e-4
bs = 128
val_dir = '../data/dataset_step_512_patch_512_scale_0.6_channels_hsv_aug_2/test/'
train_dir = '../data/dataset_step_512_patch_512_scale_0.6_channels_hsv_aug_2/train/'
model_name = 'resnet50'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class models:
    def __init__(self):
        pass

    @staticmethod
    def resnet18(input_shape=(224,224,3), classes=4):
        """
        导入ResNet18模型，模型参数是由随机初始化。
        :param input_shape: 输入图片的维度
        :param classes: 图片的类别
        :return: ResNet18模型框架
        """
        model = ResNet18(input_shape=input_shape, classes=classes)
        return model

    @staticmethod
    def resnet50(init_model_path='/cptjack/sys_software_bak/tensorflow_keras_models/models/resnet50(224).h5'):
        """
        导入ResNet50模型，模型参数是使用预训练的模型参数。在这里我在ResNet50的框架上加了一个维度为30的隐藏层。
        :param init_model_path:预训练的模型参数的路径
        :return:ResNet50模型框架
        """
        base_model = load_model(init_model_path)
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        model.summary()
        return model


today = datetime.date.today().strftime('%y%m%d')


class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_acc', mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk, self).__init__(filepath, monitor, save_best_only, mode)

    def set_model(self, model):
        super(Mycbk, self).set_model(self.single_model)


def get_callbacks(model_name, model, patience):
    """
    回调函数，用来保存模型，即损失函数等可视化操作。
    :param model_name: 模型的名字，如ResNet50, ResNet18...
    :param model: 模型框架
    :param patience: EarlyStopping， 如果val_loss在(patience)个回合没有下降，则停止学习
    :return:
    """
    base_dir = "./result/" + str(today) + "/"
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model,base_dir+ model_name + '.h5')
    file_dir = base_dir + '/log/'+str(model_name) + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                  patience=5, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger(base_dir + time.strftime('%Y_%m_%d', time.localtime(time.time())) + '_' +str(model_name) +
                       '_log.csv', separator=',', append=True)
    return [es, msave, reduce_lr, tb_log, log_cv]


def imagenet_processing(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:, :, i] -= mean[i]
        image[:, :, i] /= std[i]
    return image


if __name__ == "__main__":
    if model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'resnet18':
        model = models.resnet18()
    opt = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    trainPaths = list(paths.list_images(train_dir))
    random.seed(40)
    random.shuffle(trainPaths)
    totalTrain = len(trainPaths)
    totalVal = len(list(paths.list_images(val_dir)))

    trainAug = generators.DataGenerator(rescale=1./255, rotation_range=90,
                                        horizontal_flip=True, vertical_flip=True,
                                        preprocessing_function = imagenet_processing)

    valAug = generators.DataGenerator(rescale=1./255, preprocessing_function = imagenet_processing)

    trainGen = trainAug.flow_from_directory(train_dir,
                                            class_mode="categorical",
                                            target_size=(224, 224),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=bs)

    valGen = valAug.flow_from_directory(val_dir,
                                        class_mode="categorical",
                                        target_size=(224, 224),
                                        color_mode="rgb",
                                        shuffle=True,
                                        batch_size=bs)

    callbacks_s = get_callbacks(model_name, model, patience=5)
    H = model.fit_generator(trainGen,
                            steps_per_epoch=totalTrain // bs,
                            validation_data=valGen,
                            validation_steps=totalVal // bs,
                            epochs=num_epochs,
                            callbacks=callbacks_s,
                            verbose=1)
