# Input: 128x128 L channel
# Output: 32x32 ab channel
# Upscale: 128x128 image

import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# INTER_AREA
# INTER_CUBIC

dataset_x = []
dataset_y = []

# for folder in ['PetImages/Cat', 'PetImages/Dog'][:]:
for folder in ['LandImages']:
    for f in tqdm(os.listdir(folder)[:]):
        try:
            img = cv2.imread(os.path.join(folder, f))

            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            l_channel = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[:1]
            l_channel = np.array(l_channel)  # (1, 128, 128)

            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            ab_channel = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))[1:]
            ab_channel = np.array(ab_channel)  # (2, 32, 32)

            dataset_x.append(l_channel)
            dataset_y.append(ab_channel)

            # print(l_channel.shape, ab_channel.shape)
            #
            # plt.imshow(l_channel[0], cmap='gray')
            # plt.show()
        except Exception as e:
            pass

perm = np.array(range(len(dataset_x)))
np.random.shuffle(perm)
dataset_x = np.array([dataset_x[i] for i in perm])
dataset_y = np.array([dataset_y[i] for i in perm])

# L: 0-255
# ab: 1-255
np.save('land_x.npy', dataset_x)
np.save('land_y.npy', dataset_y)
print(np.array(dataset_x).shape, np.array(dataset_y).shape)
