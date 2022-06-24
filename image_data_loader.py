import numpy as np
import os

from torch.utils.data import Dataset
from utils import prepare_image_for_network


class ImageDataset(Dataset):
    def __init__(self, data_path: str, resize_width=256, resize_height=256):
        self.__data_path = data_path
        self.__file_names = os.listdir(self.__data_path)
        self.__resize = (resize_width, resize_height)

    def __len__(self):
        return len(self.__file_names)

    def __getitem__(self, idx):
        full_path = os.path.join(self.__data_path, self.__file_names[idx])

        return prepare_image_for_network(full_path, (256, 256))
