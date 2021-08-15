# 128x128 L channel
# 32x32 ab channel

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def combine_32x32(l, ab, index, l_float=True, ab_float=True):
    l_channel = np.load(l, allow_pickle=True)[index]
    if l_float:
        l_channel = l_channel * 255
    ab_channel = np.load(ab, allow_pickle=True)[index]
    if ab_float:
        ab_channel = ab_channel * 127 + 128

    l_channel = l_channel.astype(np.uint8)
    ab_channel = ab_channel.astype(np.uint8)

    l_channel = cv2.resize(l_channel[0], (32, 32), interpolation=cv2.INTER_AREA).reshape(32, 32, 1)
    a_channel = ab_channel[0].reshape(32, 32, 1)
    b_channel = ab_channel[1].reshape(32, 32, 1)

    lab_channel = np.concatenate((l_channel, a_channel, b_channel), axis=2)
    rgb = cv2.cvtColor(lab_channel, cv2.COLOR_LAB2RGB)
    plt.imshow(rgb)


def combine_128x128(l, ab, index, l_float=True, ab_float=True):
    l_channel = np.load(l, allow_pickle=True)[index]
    if l_float:
        l_channel = l_channel * 255
    ab_channel = np.load(ab, allow_pickle=True)[index]
    if ab_float:
        ab_channel = ab_channel * 127 + 128

    l_channel = l_channel.astype(np.uint8)
    ab_channel = ab_channel.astype(np.uint8)

    l_channel = l_channel[0].reshape(128, 128, 1)
    a_channel = cv2.resize(ab_channel[0], (128, 128), interpolation=cv2.INTER_CUBIC).reshape(128, 128, 1)
    b_channel = cv2.resize(ab_channel[1], (128, 128), interpolation=cv2.INTER_CUBIC).reshape(128, 128, 1)

    lab_channel = np.concatenate((l_channel, a_channel, b_channel), axis=2)
    rgb = cv2.cvtColor(lab_channel, cv2.COLOR_LAB2RGB)
    plt.imshow(rgb)


def gray_32x32(l, index, l_float=True):
    l_channel = np.load(l, allow_pickle=True)[index]
    if l_float:
        l_channel = l_channel * 255

    l_channel = l_channel.astype(np.uint8)
    l_channel = cv2.resize(l_channel[0], (32, 32), interpolation=cv2.INTER_AREA).reshape(32, 32)
    plt.imshow(l_channel, cmap='gray')


def gray_128x128(l, index, l_float=True):
    l_channel = np.load(l, allow_pickle=True)[index]
    if l_float:
        l_channel = l_channel * 255

    l_channel = l_channel.astype(np.uint8)
    l_channel = cv2.resize(l_channel[0], (128, 128), interpolation=cv2.INTER_CUBIC).reshape(128, 128)
    plt.imshow(l_channel, cmap='gray')


def load_l(f, index, is_float=True):
    l_channel = np.load(f, allow_pickle=True)[index]
    if is_float:
        l_channel = l_channel * 255
    return l_channel


def load_ab(f, index, is_float=True):
    ab_channel = np.load(f, allow_pickle=True)[index]
    if is_float:
        ab_channel = ab_channel * 127 + 128

    return ab_channel


def load_ab_onehot(f, index):
    ab_channel_onehot = np.load(f, allow_pickle=True)[index]
    res = ab_channel_onehot.shape[1]
    ab_channel = np.zeros((2, res, res))

    for x in range(res):
        for y in range(res):
            ab_channel[0, x, y] = np.argmax(ab_channel_onehot[:32, x, y]) * 8
            ab_channel[1, x, y] = np.argmax(ab_channel_onehot[32:, x, y]) * 8

    return ab_channel


def combine(l, ab, res):
    if ab is not None:
        l_channel = l.astype(np.uint8)
        ab_channel = ab.astype(np.uint8)

        l_channel = l_channel[0]
        if l_channel.shape[0] < res:
            l_channel = cv2.resize(l_channel, (res, res), interpolation=cv2.INTER_CUBIC)
        elif l_channel.shape[0] > res:
            l_channel = cv2.resize(l_channel, (res, res), interpolation=cv2.INTER_AREA)
        l_channel = l_channel.reshape(res, res, 1)

        a_channel = ab_channel[0]
        if a_channel.shape[0] < res:
            a_channel = cv2.resize(a_channel, (res, res), interpolation=cv2.INTER_CUBIC)
        elif a_channel.shape[0] > res:
            a_channel = cv2.resize(a_channel, (res, res), interpolation=cv2.INTER_AREA)
        a_channel = a_channel.reshape(res, res, 1)

        b_channel = ab_channel[1]
        if b_channel.shape[0] < res:
            b_channel = cv2.resize(b_channel, (res, res), interpolation=cv2.INTER_CUBIC)
        elif b_channel.shape[0] > res:
            b_channel = cv2.resize(b_channel, (res, res), interpolation=cv2.INTER_AREA)
        b_channel = b_channel.reshape(res, res, 1)

        lab_channel = np.concatenate((l_channel, a_channel, b_channel), axis=2)
        rgb = cv2.cvtColor(lab_channel, cv2.COLOR_LAB2RGB)
        plt.imshow(rgb)
    else:
        l_channel = l.astype(np.uint8)

        l_channel = l_channel[0]
        if l_channel.shape[0] < res:
            l_channel = cv2.resize(l_channel, (res, res), interpolation=cv2.INTER_CUBIC)
        elif l_channel.shape[0] > res:
            l_channel = cv2.resize(l_channel, (res, res), interpolation=cv2.INTER_AREA)

        plt.imshow(l_channel, cmap='gray')


INDEX = 0
FILE_NAME = 'result_v10'

# combine_32x32('pet_color_x.npy', 'pet_color_y.npy', INDEX, l_float=False, ab_float=False)
# combine_128x128('pet_color_x.npy', 'pet_color_y.npy', INDEX, l_float=False, ab_float=False)

# combine_32x32('pet_color_x_unshuffled.npy', 'pet_color_y_unshuffled.npy', INDEX, l_float=False, ab_float=False)
# combine_128x128('pet_color_x_unshuffled.npy', 'pet_color_y_unshuffled.npy', INDEX, l_float=False, ab_float=False)

for INDEX in range(10):
    rows = 2
    columns = 3
    fig = plt.figure(figsize=(9, 6))

    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.title('True')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), load_ab_onehot(f'{FILE_NAME}_true.npy', INDEX), 32)

    fig.add_subplot(rows, columns, 4)
    plt.axis('off')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), load_ab_onehot(f'{FILE_NAME}_true.npy', INDEX), 128)

    fig.add_subplot(rows, columns, 2)
    plt.axis('off')
    plt.title('Prediction')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), load_ab_onehot(f'{FILE_NAME}.npy', INDEX), 32)

    fig.add_subplot(rows, columns, 5)
    plt.axis('off')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), load_ab_onehot(f'{FILE_NAME}.npy', INDEX), 128)

    fig.add_subplot(rows, columns, 3)
    plt.axis('off')
    plt.title('Gray')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), None, 32)

    fig.add_subplot(rows, columns, 6)
    plt.axis('off')
    combine(load_l(f'{FILE_NAME}_gray.npy', INDEX), None, 128)

    plt.show()
