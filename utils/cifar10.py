# encoding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt


data_dir = "../data/cifar10"


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_data(mode, num=-1):
    if mode == "train":
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    else:
        files = ["test_batch"]

    d = np.empty(shape=[0, 3072], dtype="uint8")
    l = []
    for file in files:
        file = data_dir + "/" + file
        data = unpickle(file)
        d = np.concatenate((d, data[b"data"]), axis=0)
        l += data[b"labels"]
    d = np.transpose(np.reshape(d, newshape=[-1, 3, 32, 32]), axes=[0, 2, 3, 1])
    l = np.array(l)
    if num != -1:
        d = d[0: num]
        l = l[0: num]
    return d, l


def get_batches(images, labels, batch_size, shuffle=True):
    idx = np.arange(images.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[(i * batch_size): ((i + 1) * batch_size)] for i in range(int(images.shape[0] / batch_size))]
    for batch in batches:
        yield (images[batch], labels[batch])








# d, l = load_data("test")
#
# print(d.shape, l.shape)
# print(d[0] / 255.0)
# plt.imshow(d[0] / 255.0)
# plt.show()