import os.path
import os, path
import json
import scipy.misc
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
        self.current_epoch = 0
        self.batch_index = 0

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
        image_label = self.class_dict.get(image_name)

        return image_label

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        #Shuffling extracted_images and class_names
        if self.shuffle:
            permutation_indices = np.random.permutation(self.N_images)
            extracted_images = self.extracted_images[permutation_indices]
            class_names = self.class_names[permutation_indices]
        else:
            extracted_images = self.extracted_images.copy()
            class_names = self.class_names.copy()

        #Mirroring n random images that batch contains both mirrored and non-mirrored images
        if self.mirroring:
            n_elements_to_flip = np.random.randint(self.N_images)
            random_index_to_mirror = np.random.randint(0, self.N_images, n_elements_to_flip)
            for i in range(n_elements_to_flip):
                extracted_images[random_index_to_mirror[i]] = np.fliplr(extracted_images[random_index_to_mirror[i]])

        #Rotating n random images that batch contains both mirrored and non-mirrored images
        if self.rotation:
            n_elements_to_rotate = np.random.randint(self.N_images)
            rotation_val_sel = np.random.randint(0, 3, n_elements_to_rotate)
            for i in range(n_elements_to_rotate):
                if rotation_val_sel[i] == 0:
                    extracted_images[i] = rotate(extracted_images[i], 90)
                elif rotation_val_sel[i] == 1:
                    extracted_images[i] = rotate(extracted_images[i], 180)
                else:
                    extracted_images[i] = rotate(extracted_images[i], 270)

        #If last batch is smaller than others, completing batch by reusing images from beginning of training data set
        batch_indices = np.arange(self.batch_index, self.batch_index + self.batch_size)
        if np.max(batch_indices) > self.N_images:
            indices_greater_than_max_val = np.argwhere(batch_indices > self.N_images)[0]
            batch_indices[indices_greater_than_max_val] = np.arange(0, len(indices_greater_than_max_val))
        images = extracted_images[batch_indices,:]
        labels = class_names[batch_indices]
        self.batch_index = self.batch_index + self.batch_size

        #for i in range(len(images)):
        #    images[i] = resize(images[i], self.image_size)

        #Updating value of current epoch
        if max(batch_indices) == len(extracted_images) - 1:
            self.current_epoch += 1

        """
        newbatch_start = 0
        images = np.empty((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        labels = []
        for i in range(newbatch_start, newbatch_start + self.batch_size):
            images[i] = resize(extracted_images[i], self.image_size)
            if self.rotation:
                n = random.randint(1, 3)
                if n == 1:
                    images[i] = rotate(images[i], 90)
                elif n == 2:
                    images[i] = rotate(images[i], 180)
                else:
                    images[i] = rotate(images[i], 270)


            

            # string not stored in array so save each label in list
            label = self.class_name(image_list[i])
            labels.append(label)
        """

        return images, labels

    def augment(self, img):
        #
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch

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
