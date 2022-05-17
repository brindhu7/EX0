import os.path
import os, path
import json

import scipy
from scipy import ndimage, misc

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
import random


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch= 0
        self.process_once = False
        self.batch_start = 0

        # opening the JSON files - Getting the labels for all images
        f = open(self.label_path)
        self.label_data = json.load(f)
        f.close()

        # list of .npy files
        image_list = np.array(os.listdir(self.file_path))

        extracted_images = []
        self.class_names = []

        # Extracting each image and corresponding class name from file path
        for i in range(len(image_list)):
            extracted_images.append(np.load(self.file_path + "/" + image_list[i]))
            self.class_names.append(self.class_name(image_list[i]))

        self.extracted_images = np.array(extracted_images)
        self.N_images = len(image_list)

    def class_name(self, x):
        # This function returns the class name for a specific input

        image_number = x.split(".")
        image_name = self.label_data.get(image_number[0])
        #image_label = self.class_dict.get(image_name)

        return int(image_name)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases


        #If last batch is smaller than others, completing batch by reusing images from beginning of training data set
        #initializing image array
        images = np.empty((self.batch_size,self.image_size[0],self.image_size[1],self.image_size[2]),dtype='uint8')
        batch_indices = np.arange(self.batch_start,self.batch_start + self.batch_size)

        if np.max(batch_indices) > self.N_images:
            self.process_once = True
            remaining = np.max(batch_indices) - (self.N_images-1)
            batch_indices = batch_indices[:-remaining]
            previous_image_index = np.arange(remaining)
            batch_indices = np.append(batch_indices, previous_image_index)
            #if batch_indices[0]==0:
            #    self.batch_start = 0

        # extracting images, labels from the whole image array and label array respectively with batch indices
        A = self.extracted_images[batch_indices,:]
        labels = self.class_names[self.batch_start: self.batch_start + self.batch_size]

        # resizing the images
        for i in range(len(batch_indices)):
            images[i] = np.resize(A[i],(self.image_size[0],self.image_size[1],self.image_size[2]))

        # Rotating n random images that batch contains both rotated and non-rotated images

        if self.rotation:
            num = np.random.randint(self.batch_size)
            for j in range(num):
                n = random.randint(1, 3)
                if n == 1:
                    images[j] = ndimage.rotate(images[j], 90)
                elif n == 2:
                    images[j] = ndimage.rotate(images[j], 180)
                else:
                    images[j] = ndimage.rotate(images[j], 270)

        if self.mirroring:
            num_mirror = np.random.randint(self.batch_size)
            for k in range(num_mirror):
                n = random.randint(0, 1)
                if n==0:
                    images[k] = images[k][::-1,:,:]

        if self.shuffle:
            original_images = images.copy()
            original_labels = labels.copy()
            new_order = np.random.permutation(len(A))
            for j in range(len(A)):
                images[j] = original_images[new_order[j]]
                labels[j] = original_labels[new_order[j]]

        self.batch_start = self.batch_start + self.batch_size

        return images, labels

    #def augment(self, img):
        #
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image





    #return img

    def current_epoch(self):
        # return the current epoch number
        if self.process_once:
            self.epoch += 1
            self.batch_start = 0

        return self.epoch

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        fig = plt.figure()

        #Plotting
        for i in range(self.batch_size):
            plt.subplot(4,3,i+1)
            plt.imshow(images[i])
            plt.title(labels[i])
        plt.show()