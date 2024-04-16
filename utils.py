import os
import random

from PIL import Image

import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from skimage.color import rgb2gray

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from skimage.measure import block_reduce
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def load_data(path_to_data, DOWNSAMPLE_FACTOR = 7):
    '''Load 2D images and their corresponding labels
    Parameters:
    path_to_data (str): This is the path to data
    
    Returns:
    images (np.ndarray): A numpy array of shape (N, 64, 64, 3)
    labels (np.ndarray): A numpy array of shape (N)
    
    '''
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
                if DOWNSAMPLE_FACTOR != 1:
                    image_as_array = block_reduce(image_as_array, block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR, 1), func=np.max)
    
                labels.append(label)
                images.append(image_as_array)
            except:
                num_failed += 1

    print(str(num_failed) + ' images failed to load.')

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def fit_transform_PCA(X_train, y_train, variance_threshold=0.9, max_components=100):
    sc = StandardScaler()

    X_train_scaled = sc.fit_transform(X_train)

    # Find "optimal" number of components first.
    n_components = min(X_train.shape[1], max_components)
    pca = PCA(n_components = n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)
    axs[0].set(xlabel='Number of Principal Components')
    axs[0].set(ylabel='Total Explained Variance')
    axs[0].set_title('# Components vs. Total Explained Variance')

    for img_class in list(set(y_train)):
        class_filter = np.where(y_train == img_class)
        axs[1].scatter(X_train_pca[class_filter, 0], X_train_pca[class_filter, 1], label=img_class)
    axs[1].set(xlabel='Principal Component 1')
    axs[1].set(ylabel='Principal Component 2')
    axs[1].set_title('2 Component PCA')
    axs[1].legend()
    plt.show()

    # Find and apply PCA with number of components to explain maximum variance not exceeding threshold
    if cumulative_explained_variance[-1] < variance_threshold:
        n_components = len(cumulative_explained_variance)
        print(n_components, 'out of', X_train.shape[1], 'components needed to exceed a total explained variance of', cumulative_explained_variance[-1])
    else:
        n_components = np.searchsorted(cumulative_explained_variance, variance_threshold, side='right') + 1
        print(n_components, 'out of', X_train.shape[1], 'components needed to exceed a total explained variance of', variance_threshold)

    pca = PCA(n_components = n_components)
    X_train = pca.fit_transform(X_train_scaled)
    return X_train, pca, sc


def transform_PCA(pca, sc, X_val, X_test):
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_val_pca, X_test_pca


def plot_confusion_matrix(model, true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()


def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print('Accuracy:', accuracy)
    # Precision
    precision = precision_score(true_labels, predicted_labels, pos_label="wildfire")
    print('Precision:', precision)
    # Recall
    recall = recall_score(true_labels, predicted_labels, pos_label="wildfire")
    print('Recall:', recall)
    # F1 
    f1 = fbeta_score(true_labels, predicted_labels, pos_label="wildfire", beta=1, average ='binary')
    print('F1:', f1)

    f2 = fbeta_score(true_labels, predicted_labels, pos_label="wildfire", beta=2, average ='binary')
    print('F2:', f2)
    #return accuracy, precision, recall, f1

def resize_imgs(data, target=(224, 224)):
    resized_images = []
    for img_array in data:
        img = Image.fromarray(np.uint8(img_array))
        resized_img = img.resize(target)
        resized_images.append(resized_img)
    return np.array(resized_images)