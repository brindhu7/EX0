import numpy as np
import matplotlib.pyplot as plt


image = np.zeros([256, 256, 3])

red_ch = np.linspace(0, 1, 256)[np.newaxis, :]*np.ones([1, 256])

blue_ch = np.linspace(1, 0, 256)[np.newaxis, :]*np.linspace(1, 0.5, 256)[:, np.newaxis]

green_ch = np.linspace(0.5, 1, 256)[np.newaxis, :]*np.linspace(0, 1, 256)[:, np.newaxis]

image[:, :, 0] = red_ch
image[:, :, 1] = green_ch
image[:, :, 2] = blue_ch

plt.figure()
plt.imshow(image)

plt.show()
