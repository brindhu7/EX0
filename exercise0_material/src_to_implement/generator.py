import os.path
import os, path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
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

        self.batch_index = 0
        self.shuffled = False
        self.process_once = False
        self.epoch = 0

        # opening the JSON files - Getting the labels for all images
        f = open(self.label_path)
        self.label_data = json.load(f)
        f.close()

        # list of .npy files
        image_list = np.array(os.listdir(self.file_path))

        extracted_images = []
        class_names = []

        # Extracting each image and corresponding class name from file path
        for i in range(len(image_list)):
            extracted_images.append(np.load(self.file_path + "/" + image_list[i]))
            class_names.append(self.class_name(image_list[i]))
        self.extracted_images = np.array(extracted_images)
        self.class_names = np.array(class_names)
        self.N_images = len(image_list)

    def class_name(self, x):
        # This function returns the class name for a specific input

        image_number = x.split(".")
        image_name = self.label_data.get(image_number[0])
       # image_label = self.class_dict.get(image_name)

        return int(image_name)


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # Shuffling extracted_images and class_names

        extracted_images = self.extracted_images.copy()
        class_names = self.class_names.copy()

        images = np.empty((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                          dtype='uint8')

        batch_indices = np.arange(self.batch_index, self.batch_index + self.batch_size)
        if np.max(batch_indices) >= self.N_images:
            self.process_once = True
            remaining = np.max(batch_indices) - (self.N_images - 1)
            batch_indices = batch_indices[:-remaining]
            previous_image_index = np.arange(remaining)
            batch_indices = np.append(batch_indices, previous_image_index)




        A = extracted_images[batch_indices, :]
        labels = class_names[batch_indices]

        for i in range(len(batch_indices)):
            images[i] = np.resize(A[i], (self.image_size[0], self.image_size[1], self.image_size[2]))

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
                if n == 0:
                    images[k] = images[k][::-1, :, :]



        if self.shuffle:

            image_array = images.copy()
            label_array = labels.copy()
            permutation_indices = np.random.permutation(len(images))
            images = image_array[permutation_indices]
            labels = label_array[permutation_indices]

        self.batch_index = self.batch_index + self.batch_size

        return images,labels

    def current_epoch(self):
        # return the current epoch number
        if self.process_once:
            self.epoch += 1

        return self.epoch