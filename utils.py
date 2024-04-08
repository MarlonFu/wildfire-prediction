import os
import random

from PIL import Image

import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def load_data(path_to_data, DOWNSAMPLE_FACTOR = 7):
    '''Load 2D images and their corresponding labels
    Parameters:
    path_to_data (str): This is the path to data
    
    Returns:
    images (np.ndarray): A numpy array of shape (N, 64, 64, 3)
    labels (np.ndarray): A numpy array of shape (N)
    
    '''
    ## load images and labels
    # FILL IN CODE HERE #
    images = []
    labels = []

    num_failed = 0

    for label in os.listdir(path_to_data):
        if label.startswith('.'):
            continue;
        
        label_folder_path = os.path.join(path_to_data, label)
        
        for image in os.listdir(label_folder_path):
            try:
                loaded_image = load_img(os.path.join(label_folder_path, image))
                image_as_array = img_to_array(loaded_image)
                # Downsample image
                image_as_array = block_reduce(image_as_array, block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR, 1), func=np.max)
    
                labels.append(label)
                images.append(image_as_array)
            except:
                num_failed += 1

    print(str(num_failed) + ' images failed to load.')

    images = np.array(images)
    labels = np.array(labels)

    return images, labels