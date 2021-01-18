# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from skimage.transform import resize
import build_model


ORIG_ROW = 420
ORIG_COL = 580


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


def predict_mask(model, imgs, fnames):
    """Predict masks for test images.

    Args:
        model: best trained model.
        imgs: float ndarray of images
        fnames: list of names of the images

    Returns: DataFrame of image names and encoded mask predictions

    """
    pred = pd.DataFrame([], columns=['img', 'pixels'])
    
    for idx, fname in enumerate(fnames):
        img = np.expand_dims(imgs[idx], axis=0)
        mask_pred = model.predict(img)
        mask_pred = resize(mask_pred[0,:,:,0], (ORIG_ROW, ORIG_COL))
        mask_pred = np.rint(mask_pred)
        print(fname)
        pred = pred.append(
            {'img':fname, 
             'pixels':run_len_encoding(mask_pred)}, ignore_index=True)
        
    return pred


def predict_masks(test_imgs_npy, best_weight_fname, 
                  pred_mask_fname="test_masks_pred.csv"):
    # Load images
    test_imgs = np.load(test_imgs_npy) #"test_imgs.npy")
    
    # Load model and weight
    model = build_model.unet()
    model.load_weights(best_weight_fname) #"best_weight.h5")
    
    # Predict masks for test data
    test_fnames = os.listdir(os.path.join('..', '..', 'data', 'raw', 'test'))
    test_masks_pred = predict_mask(model, test_imgs, test_fnames)
    test_masks_pred.to_csv(pred_mask_fname, index=False)
