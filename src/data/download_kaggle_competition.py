# -*- coding: utf-8 -*-
import os
import kaggle

def download_dataset(competition='ultrasound-nerve-segmentation', 
                     output_filepath=os.path.join('..', '..', 'data', 'raw')):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(competition, path=output_filepath)