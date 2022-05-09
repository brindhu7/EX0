import os.path
import os,path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize,rotate
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

        #opening the JSON files - Getting the labels for all images
        f = open(self.label_path)
        self.label_data =  json.load(f)
        f.close()

        # list of .npy files
        image_list = os.listdir(self.file_path)

        newbatch_start = 0
        images = np.empty((self.batch_size,self.image_size))

        for i in range(newbatch_start,newbatch_start+self.batch_size):
            images[i] = resize(np.load(self.file_path + image_list[i]),self.image_size)
            if self.rotation == True:
                n = random.randint(1, 3)
                if n==1:
                    images[i] = rotate(images[i],90)
                elif n==2:
                    images[i] = rotate(images[i],180)
                else:
                    images[i] = rotate(images[i],270)
            if self.mirroring == True:
                images[i] = np.flip(images[i], 1)

            labels = []
            # string not stored in array so save each label in list
            label = self.class_name(image_list[i])
            labels = np.append(labels, label)










        return images, labels


    def augment(self,img):
        #
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image


        return img

    def current_epoch(self):
        # return the current epoch number
        return 0


    def show(self):
    # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.


        images,labels = self.next()
