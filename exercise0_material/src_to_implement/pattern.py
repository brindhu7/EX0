import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self,resolution,tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        assert self.resolution % (2 * self.tile_size) == 0
        #Finding Number tiles required for the given resolution
        no_of_tiles = self.resolution // self.tile_size
        #creating 0's and 1's tile
        zero_horizontal = np.zeros((self.tile_size,self.tile_size))
        ones_horizontal = np.ones((self.tile_size,self.tile_size))
        #Creating first 4 blocks using stack
        hs0 = np.hstack((zero_horizontal,ones_horizontal))
        hs1 = np.hstack((ones_horizontal,zero_horizontal))
        first_tile_array = np.vstack((hs0,hs1))
        #Repeating the pattern
        self.output = np.tile(first_tile_array,(no_of_tiles//2,no_of_tiles//2))
        return self.output.copy()


    def show(self):
        plt.imshow(self.output,cmap= 'Blues_r')
        plt.show()

class Circle:
    def __init__(self,resolution,radius,position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        # creating X,Y  axis for meshgrid
        x_axis = np.linspace(0,self.resolution,self.resolution)
        y_axis = np.linspace(0,self.resolution,self.resolution)
        #Getting the centre position
        circle_centre_X , circle_centre_Y = self.position
        # Getting X,Y coordinates
        aa,bb = np.meshgrid(x_axis,y_axis)
        #Circle equation
        self.output = ((aa - circle_centre_X) ** 2 + (bb - circle_centre_Y) ** 2) <= self.radius ** 2
        return self.output.copy()

    def show(self):
        plt.imshow(self.output,cmap='Blues_r')
        plt.show()

class Spectrum:
    def __init__(self,resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):

        image_array1 = np.linspace(0,1,self.resolution)

        image_array2 = np.transpose(image_array1)
        image_array3 = np.linspace(1,0,self.resolution)

        self.output = np.stack((image_array1,image_array2,image_array3))

        return self.output.copy()

    def show(self):

        plt.imshow(self.output)
        plt.show()
