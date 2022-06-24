import numpy as np
import torch

from PIL import Image
from typing import Tuple
from skimage import color


def load_image(image_path: str, new_size: Tuple[int, int]) -> np.ndarray:
    img: Image.Image = Image.open(image_path)
    img: Image.Image = img.resize(new_size)

    return np.asarray(img)


def convert_image_to_lab(image: np.ndarray):
    return color.rgb2lab(image)


def convert_lab_image_to_tensor(image):
    return torch.Tensor(image)


def prepare_image_for_network(image_path: str, new_size: Tuple[int, int]):
    image = load_image(image_path, new_size)
    image = convert_image_to_lab(image)
    image = convert_lab_image_to_tensor(image)

    return image