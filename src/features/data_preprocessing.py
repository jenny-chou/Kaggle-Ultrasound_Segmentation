# -*- coding: utf-8 -*-
import os
import numpy as np
from zipfile import ZipFile
from skimage.io import imread
from skimage.transform import resize


RESIZED_ROW = 128
RESIZED_COL = 128


def unzip_dataset(zipfile_fpath, destination_path):
    """Unzip Kaggle dataset.

    Args:
        zipfile_fpath: path and name of Kaggle dataset zipfile
        destination_path: unzip dataset to destination
    """
    with ZipFile(zipfile_fpath, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(path=destination_path)
       
    print("Unzipped dataset")
    
    
def list_fpaths(data_raw_path):
    """List relative paths of all train and test images.

    Args:
        data_raw_path: common prefix of all dataset

    Returns: list of three lists: paths of train images, paths of train mask 
             images, and paths of test images
    """
    train_fpaths = os.listdir(os.path.join(data_raw_path, 'train'))
    test_fpaths = os.listdir(os.path.join(data_raw_path, 'test'))
    
    train_img_fpaths = [os.path.join(data_raw_path, 'train', fname) \
                        for fname in train_fpaths if 'mask' not in fname]
        
    train_mask_fpaths = [os.path.join(data_raw_path, 'train', fname) \
                         for fname in train_fpaths if 'mask' in fname]
    
    test_img_fpaths = [os.path.join(data_raw_path, 'test', fname) \
                         for fname in test_fpaths]
        
    print("Listed all file paths")
        
    return [train_img_fpaths, train_mask_fpaths, test_img_fpaths]


def resize_imgs(fpaths):
    """Resize images.

    Args:
        fpaths: paths of images

    Returns: resized images
    """
    resized_imgs = np.ndarray([len(fpaths), RESIZED_ROW, RESIZED_COL, 1])
    for idx in range(len(fpaths)):
        orig_img = np.array(imread(fpaths[idx]), dtype=np.uint8)
        resized_imgs[idx] = resize(orig_img, [RESIZED_ROW, RESIZED_COL, 1])
        
        # print progress
        if idx % 120 == 0:
            print(f'Resized {idx} images')
    
    print("Resized all images")
    
    return resized_imgs


def data_preprocessing(zip_fpath, data_processed_path):
    """Resize images.

    Args:
        fpaths: ndarray containing images

    Returns: resized images in float ndarray
    """
    # unzip dataset
    data_raw_path = os.path.dirname(zip_fpath)
    unzip_dataset(zip_fpath, data_raw_path)

    # find all images
    all_fpaths = list_fpaths(data_raw_path)
    npy_names = ["train_imgs.npy", "train_masks.npy", "test_imgs.npy"]
    
    # resize images and save to npy form
    for fpaths, npy_name in zip(all_fpaths, npy_names):
        npy_fpath = os.path.join(data_processed_path, npy_name)
        resized_imgs = resize_imgs(fpaths)        
        np.save(npy_fpath, resized_imgs)