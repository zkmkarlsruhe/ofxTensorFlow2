from config import *
from os.path import join, dirname
from tensorflow.keras.utils import get_file


def download_existing_dataset(dataset_url, dataset_name):
    '''Download Dataset
    Params:
        dataset_url     -> Url to the dataset
        dataset_name    -> Name of the dataset
    '''
    file_name = dataset_url.split('/')[-1]
    path = get_file(
        file_name,
        origin=dataset_url,
        extract=True
    )
    path = join(dirname(path), dataset_name)
    return path


