from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger,
    TerminateOnNaN,
    ModelCheckpoint
)
import _parameter as p
import network
import os
import keras.backend as K
import tensorflow as tf

def get_generator(batch_size, dirname):
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
        preprocessing_function=None,
        data_format="channels_last", # channels_first or channels_last=(samples, height, width, channels)
        validation_split=0.0,
        dtype=None)
    generator = data_generator.flow_from_directory(
        dirname, # img/train
        target_size=(p.IMG_ROWS, p.IMG_COLS),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
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

    
def run(args):
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=2)
    early_stopper = EarlyStopping(min_delta=0, patience=1)
    csv_logger = CSVLogger("csv/logger.csv", separator=",", append=False)
    nan_terminater = TerminateOnNaN()
    check_pointer = ModelCheckpoint("models/best_model.h5",
                                    monitor="val_acc",
                                    #monitor="mymetric",
                                    mode="max",
                                    save_best_only=True)


    model = network.ResnetBuilder.build_resnet_18((p.CHANNELS, p.IMG_ROWS, p.IMG_COLS), p.CLASSES)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer='sgd',
                  metrics=['accuracy']
                  #metrics=[mymetric]
                  )
        

    train_generator = get_generator(batch_size=32, dirname="img/train")
    test_generator = get_generator(batch_size=32, dirname="img/test")
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        verbose=1,
        callbacks=[lr_reducer, early_stopper, csv_logger, nan_terminater, check_pointer],
        validation_data=test_generator,
        validation_steps=len(test_generator),
        class_weight=None,
        max_queue_size=10,
        workers=args.workers,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0
        )
    
