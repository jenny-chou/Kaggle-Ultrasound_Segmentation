# -*- coding: utf-8 -*-
"""Build segmentation model to identify nerve structures in ultrasound images of the neck.
U-Net model is used and evaluated on the dice coefficient.
"""


import os
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, \
    Concatenate, MaxPool2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import gc
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt


ORG_IMG_ROW = 420
ORG_IMG_COL = 580
RESIZE_IMG_ROW = 128
RESIZE_IMG_COL = 128


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
    return 1-dice_coef(y_true, y_pred)

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
    inputs = Input(shape=(RESIZE_IMG_ROW, RESIZE_IMG_COL, 1))
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

    # plot model architecture
    plot_model(model, show_shapes=True)
    
    return model

def load_imgs(fnames):
    """Load images with with given file names.

    Args:
        fnames: file names/paths of images

    Returns: images in uint8 type ndarray

    """
    imgs = np.ndarray([len(fnames), ORG_IMG_ROW, ORG_IMG_COL], dtype=np.uint8)
    for idx, fname in enumerate(fnames):
        imgs[idx] = np.array(imread(fname), dtype=np.uint8)
        if idx%200==0:
            print(idx)
    
    return imgs

def resize_imgs(imgs):
    """Resize given images.

    Args:
        imgs: ndarray containing images

    Returns: resized images in float ndarray

    """
    resized_imgs = np.ndarray([len(imgs), RESIZE_IMG_ROW, RESIZE_IMG_COL, 1])
    for idx in range(len(imgs)):
        resized_imgs[idx] = resize(imgs[idx], [RESIZE_IMG_ROW, RESIZE_IMG_COL, 1])
    
    return resized_imgs

def load_data(train_fnames, test_fnames):
    """Load train and test images and masks.

    Args:
        train_fnames: list of training image names
        test_fnames: list of testing image names

    Returns:
        imgs: ndarray of resized training images
        masks: ndarray of resized training masks
        test_imgs: ndarray of resized test images

    """
    imgs = resize_imgs(load_imgs(["train/"+fname+".tif" for fname in train_fnames]))
    masks = resize_imgs(load_imgs(["train/"+fname+"_mask.tif" for fname in train_fnames]))
    test_imgs = resize_imgs(load_imgs(["test/"+fname+".tif" for fname in test_fnames]))

    return imgs, masks, test_imgs

def train_valid_split(imgs, masks, valid_size):
    """Split given images and masks to training set and validation set.
    Some patients' ultrasound images might be continuous. Instead of picking images
    randomly, evenly pick images from image and mask dataset for validation.

    Args:
        imgs: ndarray of images
        masks: ndarray of masks
        valid_size: percentage of training dataset to be validation dataset

    Returns:
        train_imgs: ndarray of training images
        train_masks: ndarray of training masks
        valid_imgs: ndarray of validation images
        valid_masks: ndarray of validation masks

    """
    valid_size = int(1/valid_size)
    valid_idx = np.arange(0, len(imgs), valid_size)
    valid_imgs = imgs[valid_idx]
    valid_masks = masks[valid_idx]
    train_imgs = np.delete(imgs, valid_idx, axis=0)
    train_masks = np.delete(masks, valid_idx, axis=0)
    
    return train_imgs, train_masks, valid_imgs, valid_masks

def run_len_encoding(img):
    """Compress image using run-length encoding.

    Args:
        img: binary array of image

    Returns: string of encoded image

    """
    position = 0
    pixel = 0
    count_one = 0
    previous = 0
    encoded_img = []
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            position += 1
            pixel = img[row, col]
            if pixel == 1:
                if pixel != previous:
                    encoded_img.append(str(position))
                count_one += 1
            elif pixel == 0 and pixel != previous:
                encoded_img.append(str(count_one))
                count_one = 0
            
            previous = pixel
    
    return " ".join(encoded_img)

def predict_test(model, imgs, fnames):
    """Predict masks for test images.

    Args:
        model: best trained model.
        imgs: float ndarray of images
        fnames: list of names of the images

    Returns: DataFrame of image names and encoded mask predictions

    """
    pred = pd.DataFrame([], columns=['img', 'pixels'])
    
    for idx in range(len(imgs)):
        fname = fnames[idx]
        img = np.expand_dims(imgs[idx], axis=0)
        mask_pred = model.predict(img)
        mask_pred = resize(mask_pred[0,:,:,0], (ORG_IMG_ROW, ORG_IMG_COL))
        mask_pred = np.rint(mask_pred)
        print(fname)
        pred = pred.append(
            {'img':fname, 
             'pixels':run_len_encoding(mask_pred)}, ignore_index=True)
        
    return pred


def train_n_pred():
    """Train U-Net model with training and validation datasets
    and make prediction on test images.
    """

    # List file names   
    filenames = pd.read_csv("train_masks.csv")
    
    train_fnames = ["{}_{}".format(subj, img) for subj, img in filenames[["subject", "img"]].values]
    test_fnames = [fname.split(".")[0] for fname in os.listdir("test/")]
    test_fnames = np.sort([int(fname) for fname in test_fnames])
    
    # Load and resize images (comment this section out if data had been saved to npy files)
    imgs, masks, test_imgs = load_data(train_fnames, test_fnames)
    
    # Save as numpy data file (comment this section out if data had been saved to npy files)
    np.save("imgs.npy", imgs)
    np.save("masks.npy", masks)
    np.save("test_imgs.npy", test_imgs)
    
    # Load saved images
    imgs = np.load("imgs.npy")
    masks = np.load("masks.npy")
    test_imgs = np.load("test_imgs.npy")
    
    # Split training and validation dataset
    valid_size = 0.2
    train_imgs, train_masks, valid_imgs, valid_masks = \
        train_valid_split(imgs, masks, valid_size)
    del imgs, masks
    gc.collect()
            
    # Load Unet model
    model = unet()
    
    # Set callback to save best model
    checkpoint = ModelCheckpoint("best_weight.h5", monitor='val_accuracy', 
                                 save_best_only=True, save_weights_only=True)
    
    # Train model
    epochs = 30
    batch_size = 32
    history = model.fit(train_imgs, train_masks, epochs=epochs, 
                        batch_size=batch_size, shuffle=True, random_state=7,                        
                        validation_data=(valid_imgs, valid_masks), 
                        callbacks=[checkpoint])
    
    # Plot train and validation loss
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.legend(['val_loss', 'loss'])
    
    # Load best weight
    model.load_weights("best_weight.h5")
    
    # Predict masks for test data
    test_masks_pred = predict_test(model, test_imgs, test_fnames)
    test_masks_pred.to_csv("test_masks_pred.csv", index=False)


if __name__ == '__main__':
    train_n_pred()
    