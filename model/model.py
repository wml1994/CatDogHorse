from keras import regularizers
from keras import backend as bk
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization


class ImageModel:
    @staticmethod
    def build(width,heigth,classes,depth=3):
        '''

        :param width:
        :param heigth:
        :param classes:
        :param depth:
        :return:
        '''
        model = Sequential()
        if bk.image_data_format()=='channels_first':
            shape=(depth,width,heigth)
        else:
            shape=(width,heigth,depth)
        model.add(Conv2D(20,(3,3),padding='same',input_shape=shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(strides=(2,2)))
        #---------------------------------------------------------------------
        model.add(Conv2D(30,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(strides=(2,2)))
        #---------------------------------------------------------------------
        model.add(Conv2D(50,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(strides=(2,2)))
        model.add(Dropout(0.5))
        #---------------------------------------------------------------------
        model.add(Flatten())
        model.add(Dense(250))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        #---------------------------------------------------------------------
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model