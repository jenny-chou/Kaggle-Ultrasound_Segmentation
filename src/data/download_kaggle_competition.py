# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import kaggle

@click.command()
@click.option('--competition', prompt='Enter Kaggle competition name')
@click.option('--output_filepath', default='../../data/raw')
def main(competition):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(competition, path=output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('Downloaded dataset for Kaggle competition: %s to %s', competition, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
