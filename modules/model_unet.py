import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Input, Model
from keras.layers import Conv2D, concatenate, MaxPooling2D, Activation, Flatten
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose
from keras.losses import binary_crossentropy

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def sensitivity(y_true, y_pred):
    true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.epsilon())

def specificity(y_true, y_pred):
    true_negatives = keras.sum(keras.round(keras.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = keras.sum(keras.round(keras.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.epsilon())

def bg_acc(y_true,y_pred):
    # Signal
    pred_bg = np.where(y_true==1,y_pred,-1)
    pred_bg = np.where(y_true==0,y_pred,1)
    print(type(pred_bg))
    num_bg = 1#(y_true==0).sum()
    num_pred_bg = 1#(pred_true>0.5 and pred_true>=0).sum()
    return num_pred_bg*1.0/num_bg

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True, num_classes=1, mode="bce_dice",lr=0.1):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    if mode == "dice_coef":
        model.compile(optimizer=keras.optimizers.Adam(lr=lr),loss=bce_dice_loss, metrics=[dice_coef])
    if mode == "binary_crossentropy":
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss="binary_crossentropy", metrics=[dice_coef])
    if mode == "mean_squared_error":
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss="mean_squared_error", metrics=[dice_coef])
    return model
