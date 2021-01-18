# -*- coding: utf-8 -*-
import numpy as np
import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import gc


def split_train_valid(imgs, masks, valid_ratio):
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
    data = {}
    valid_size = int(1/valid_ratio)
    valid_idx = np.arange(0, len(imgs), valid_size)
    data['valid_imgs'] = imgs[valid_idx]
    data['valid_masks'] = masks[valid_idx]
    data['train_imgs'] = np.delete(imgs, valid_idx, axis=0)
    data['train_masks'] = np.delete(masks, valid_idx, axis=0)
    
    return data


def load_train_valid_data(train_imgs_npy, train_masks_npy, valid_ratio):
    # Load images
    imgs = np.load(train_imgs_npy) #"imgs.npy")
    masks = np.load(train_masks_npy) #"masks.npy")
    
    # Split training and validation dataset
    data = split_train_valid(imgs, masks, valid_ratio)
        
    # Delete variable for space
    del imgs, masks
    gc.collect()
    
    return data


def set_callbacks(best_weight_fname):
    # Set callback to save best model
    checkpoint = ModelCheckpoint(best_weight_fname, monitor='val_accuracy', 
                                 save_best_only=True, save_weights_only=True)
    
    # Set callback to stop early 
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=3, 
                                  mode='auto', restore_best_weights=False)
    
    return [checkpoint, earlystopping]


def train_model(train_imgs_npy, train_masks_npy, valid_ratio=0.2,
                best_weight_fname="best_weight.h5"):
    # Load training and validation data
    data = load_train_valid_data(train_imgs_npy, train_masks_npy, valid_ratio)
    
    # Load model
    model = build_model.unet()
    
    # Set callbacks
    callbacks = set_callbacks(best_weight_fname)
    
    # Train model
    epochs = 100
    batch_size = 32
    history = model.fit(data['train_imgs'], data['train_masks'], 
                        epochs=epochs, batch_size=batch_size, shuffle=True,                        
                        validation_data=(data['valid_imgs'], 
                                         data['valid_masks']), 
                        callbacks=callbacks)
    
    return history