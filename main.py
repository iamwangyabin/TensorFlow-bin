import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    train=open('G:\\pytext\\reco\\datasets\\train.txt')
    mape={}
    labels = []
    images = []
    dirpath,dirname,find_walk=os.walk('G:\\pytext\\reco\\datasets\\train')
    for line in train.readlines():
        k=line.strip().split()
        mape[k[0]]=k[1]
    for root,dirs,files in os.walk('G:\\pytext\\reco\\datasets\\train'):
        for file in files:
            file_path=os.path.join(root,file)
            images.append(skimage.data.imread(file_path))
            labels.append(int(mape[file]))
    return images, labels


def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(10, 12, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


for image in images:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

def display_label_images(images, labels,label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1
    k=[k for k,v in enumerate(labels) if v==label]
    for m in k:
        image=images[m]
        plt.subplot(7, 7, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]