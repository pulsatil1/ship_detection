import gc
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import config
from functions import losses, generator

# Base Model


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


def unet(input_size, upsample):
    input_img = layers.Input(input_size, name='RGB_Input')

    pp_in_layer = input_img

    if config.NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(config.NET_SCALING)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(config.GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu',
                       padding='same')(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if config.NET_SCALING is not None:
        d = layers.UpSampling2D(config.NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    return seg_model


def create_model():

    train = os.listdir(config.TRAIN_DIR)
    test = os.listdir(config.TEST_DIR)

    masks = pd.read_csv(os.path.join(
        config.BASE_DIR, 'train_ship_segmentations_v2.csv'))

    # We stratify by the number of boats appearing so we have nice balances in each set
    masks['ships'] = masks['EncodedPixels'].map(
        lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg(
        {'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(
        lambda x: 1.0 if x > 0 else 0.0)
    masks.drop(['ships'], axis=1, inplace=True)

    # Here we undersample the empty images to get a better balanced group with more ships to try and segment
    balanced_train_df = unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(config.SAMPLES_PER_GROUP) if len(x) > config.SAMPLES_PER_GROUP else x)

    # Split & Image generators
    train_ids, valid_ids = train_test_split(balanced_train_df,
                                            test_size=0.2,
                                            stratify=balanced_train_df['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    train_gen = generator.make_image_gen(train_df)
    aug_gen = generator.create_aug_gen(train_gen)
    t_x, _ = next(aug_gen)

    valid_x, valid_y = next(generator.make_image_gen(
        valid_df, config.VALID_IMG_COUNT))

    gc.enable()
    gc.collect()

    if config.UPSAMPLE_MODE == 'DECONV':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    seg_model = unet(t_x.shape[1:], upsample)

    # callbacks setting
    weight_path = "weights/{}_weights.best.hdf5".format('seg_model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(
        monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=20)
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    seg_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses.dice_loss,
                      metrics=['binary_accuracy'])

    step_count = min(config.MAX_TRAIN_STEPS,
                     train_df.shape[0]//config.BATCH_SIZE)

    seg_model.fit(aug_gen,
                  steps_per_epoch=step_count,
                  epochs=100,
                  validation_data=(valid_x, valid_y),
                  callbacks=callbacks_list,
                  workers=1
                  )

    return seg_model


def create_fullres_model(seg_model):
    '''Complement the model so it can scale itself'''
    if config.IMG_SCALING is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(
            config.IMG_SCALING, input_shape=(None, None, 3)))
        fullres_model.add(seg_model)
        fullres_model.add(layers.UpSampling2D(config.IMG_SCALING))
    else:
        fullres_model = seg_model

    return fullres_model


if __name__ == '__main__':
    seg_model = create_model()
    seg_model.save('seg_model.h5')
    fullres_model = create_fullres_model(seg_model)
    fullres_model.save('fullres_model.h5')
