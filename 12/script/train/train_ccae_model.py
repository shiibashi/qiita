from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger,
    TerminateOnNaN,
    ModelCheckpoint
)

import numpy

import os
import cv2
DIR_PATH = os.path.dirname(__file__)
os.chdir("{}/..".format(DIR_PATH)) # script/ に移動


class CandlestickEncoder(object):
    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.model = load_model(model_filepath)
        self.encoder = Model(inputs=self.model.input,
                             outputs=self.model.get_layer(name='embedding').output
                             )
        self.encoder_feature_length = self.model.get_layer(name="embedding").output.get_shape().as_list()[1]
        s = self.model.input.get_shape().as_list() # [None, x, x, c]
        self.shape = (s[1], s[2], s[3])

    def predict_encoder_decoder(self, img_arr):
        return self.model.predict(img_arr)
    
    def predict_encoder(self, img_arr):
        features = self.encoder.predict(img_arr)
        return features

    def to_int(self, arr):
        return arr.astype("uint8")


class CCAE(object):
    def __init__(self, with_volume=False):
        if with_volume:
            shape = (256, 256, 1)
            model = CCAE_256(input_shape=shape)
        else:
            shape = (64, 64, 1)
            model = CCAE_64(input_shape=shape)
        self.shape = shape
        self.target_size = (shape[0], shape[1])
        self.model = model

def CCAE_256(input_shape):
    model = Sequential()
    pad3 = 'same'
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 5, strides=2, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=128, name='embedding'))
    model.add(Dense(units=128*16, name='dense'))
    model.add(BatchNormalization())
    model.add(Reshape((4, 4, 128)))

    #model.summary()
    model.add(Conv2DTranspose(64, 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, 5, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, 5, strides=2, padding='same', name='deconv1'))
    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv_out'))
    #model.summary()
    return model


def CCAE_64(input_shape):
    model = Sequential()
    pad3 = 'same'
    model.add(Conv2D(32, 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=32, name='embedding'))
    model.add(Dense(units=32*16, name='dense'))
    model.add(BatchNormalization())
    model.add(Reshape((2, 2, 128)))

    #model.summary()
    model.add(Conv2DTranspose(64, 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, 5, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv_out'))
    #model.summary()
    return model


def run_with_generator(with_volume=False):
    os.makedirs("data/ccae", exist_ok=True)
    ccae = CCAE(with_volume=with_volume)
    model = ccae.model
    model.summary()
    plot_model(model, to_file="data/ccae/candlestick_cae_model.png", show_shapes=True)
    model.compile(optimizer="adam", loss="mse")
    
    csv_logger = CSVLogger("data/ccae/logger.csv", separator=",", append=False)
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=10)
    early_stopper = EarlyStopping(min_delta=0.00001, patience=10)
    nan_terminater = TerminateOnNaN()
    check_pointer = ModelCheckpoint("data/ccae/best_model.h5",
                                    monitor="val_loss",
                                    mode="min",
                                    save_best_only=True)
    train_generator = get_generator(batch_size=8, dirname="data/img/train", target_size=ccae.target_size)
    test_generator = get_generator(batch_size=8, dirname="data/img/test", target_size=ccae.target_size)
    #train_generator = get_generator(batch_size=128, dirname="data/img/train", target_size=ccae.target_size)
    #test_generator = get_generator(batch_size=128, dirname="data/img/test", target_size=ccae.target_size)
    print(len(train_generator))
    print(len(test_generator))
    if "best_model.h5" in os.listdir("data/ccae"):
        print("load best_model")
        model = load_model("data/ccae/best_model.h5")
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10000,
        verbose=1,
        callbacks=[lr_reducer, early_stopper, csv_logger, nan_terminater, check_pointer],
        validation_data=test_generator,
        validation_steps=len(test_generator),
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0
        )

def get_generator(batch_size, dirname, target_size):
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,
        width_shift_range=0.,
        height_shift_range=0.,
        brightness_range=None,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocess,
        data_format="channels_last", # channels_first or channels_last=(samples, height, width, channels)
        validation_split=0.0,
        dtype=None)
    generator = data_generator.flow_from_directory(
        dirname, # img/train
        target_size=target_size,
        color_mode="grayscale",
        classes=None,
        class_mode='input',
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest'
    )
    return generator

def preprocess(x):
    x /= 255.0
    return x