# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, \
    Concatenate, MaxPool2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K


RESIZED_ROW = 128
RESIZED_COL = 128
CHANNEL = 1


def dice_coef(y_true, y_pred):
    """Dice coefficient as metrics.

    Args:
        y_true: true 2D mask
        y_pred: predict 2D mask

    Returns: scalar to evaluate prediction

    """
    smooth = 1
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    summation = K.sum(y_true_flat) + K.sum(y_pred_flat)
    return (2*intersect + smooth) / (summation + smooth)


def dice_coef_loss(y_true, y_pred):
    """Dice coefficient as loss.

    Args:
        y_true: true 2D mask
        y_pred: predict 2D mask

    Returns: scalar to evaluate training

    """
    return -dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    """Intersection over union as metrics.

    Args:
        y_true: true 2D mask
        y_pred: predict 2D mask

    Returns: scalar to evaluate prediction

    """
    smooth = 1
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersect
    return (intersect + smooth) / (union + smooth)


def unet():
    """U-Net model.

    Returns: compiled U-Net model.

    """
    inputs = Input(shape=(RESIZED_ROW, RESIZED_COL, CHANNEL))
    norm = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(norm)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    
    pool2 = MaxPool2D((2,2))(conv1)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool2)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    
    pool3 = MaxPool2D((2,2))(conv2)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool3)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    
    pool4 = MaxPool2D((2,2))(conv3)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool4)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
    
    pool5 = MaxPool2D((2,2))(conv4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool5)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)
    
    tran6 = Conv2DTranspose(512, (2,2), strides=(2,2))(conv5)
    conc6 = Concatenate(axis=3)([conv4, tran6])
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conc6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    
    tran7 = Conv2DTranspose(256, (2,2), strides=(2,2))(conv4)
    conc7 = Concatenate(axis=3)([conv3, tran7])
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conc7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    
    tran8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv3)
    conc8 = Concatenate(axis=3)([conv2, tran8])
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conc8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    
    tran9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
    conc9 = Concatenate(axis=3)([conv1, tran9])
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conc9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss=dice_coef_loss,
                  metrics=['accuracy', iou, dice_coef])

    # Plot model architecture
    plot_model(model, show_shapes=True)
    
    return model