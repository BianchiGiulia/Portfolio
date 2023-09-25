"""
Author: Giulia Bianchi
Matr.Nr.: k11819746
"""

import numpy as np

from torchvision import transforms
from PIL import Image


def img_preprocess(image_array, border_x: tuple, border_y: tuple):

    # raise err if image_array is not a numpy array or not 2D
    if isinstance(image_array, np.ndarray)== False or image_array.ndim != 2:
        raise NotImplementedError()

    try:
        # The values in border_x and border_y are not convertible to int objects
        border_x = int(border_x[0]), int(border_x[1])
        border_y = int(border_y[0]), int(border_y[1])

        # The values in border_x and border_y are smaller than 1.
        if border_x[0] < 5 or border_x[1] < 5 or border_y[0] < 5 or border_y[1] < 5:
            raise ValueError()

        # The shape of the remaining known image pixels would be smaller than (16, 16).
        if (image_array.shape[0] - sum(border_x) < 16) or (image_array.shape[1] - sum(border_y) < 16):
            raise ValueError()

    except ValueError:
        raise ValueError()

    # same shape and datatype
    input_array = image_array.copy()
    input_array[:border_x[0], :] = 0
    input_array[:, :border_y[0]] = 0
    input_array[-border_x[1]:, :] = 0
    input_array[:, -border_y[1]:] = 0

    #known_array = image_array, 0 for border, 1 for know pixels
    known_array = np.zeros_like(image_array)
    known_array[border_x[0]: input_array.shape[0] - border_x[1], border_y[0]:input_array.shape[1] - border_y[1]] = True

    #target_array= 1D with only values of border in order
    target_array = image_array[known_array == False].flatten()

    return input_array, known_array, target_array



def resize_(filename, im_shape = 90):

  resize_transforms = transforms.Compose([
  transforms.Resize(size=im_shape),
  transforms.CenterCrop(size=(im_shape, im_shape)),
  ])

  image = Image.open(filename)
  image = resize_transforms(image)

  return image