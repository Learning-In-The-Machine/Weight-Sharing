from __future__ import print_function
import json
import keras
import numpy as np

from scipy.spatial.distance import cosine
from keras.preprocessing.image import ImageDataGenerator


class MeasureDistance(keras.callbacks.Callback):
    def __init__(self):
        super(MeasureDistance, self).__init__()
        self.distance = {
            # single list
            '1': [],
            '4': [],
        }
        self.cosine = {
            # single list
            '1': [],
            '4': [],
        }

    def avg_dist_window(self, layer_weights, window_size=3):
        # init trackers
        dist = 0;
        total = 0;
        cos_sim = 0
        # specify start and end
        start = window_size;
        end = layer_weights.shape[0] - window_size

        # iterate through width and height
        for cur_i in range(start, end):
            for cur_j in range(start, end):
                # grab current kernel of interest
                current_weight = layer_weights[cur_i][cur_j]
                # iterate through sub window
                for i in range(-window_size, window_size + 1):
                    for j in range(-window_size, window_size + 1):
                        if abs(i) != window_size and abs(j) != window_size: continue
                        # this comparison will always be zero
                        if i == 0 and j == 0: continue
                        # compute euclidean distance
                        dist += np.linalg.norm(
                            current_weight - layer_weights[cur_i + i][cur_j + j]
                        )
                        cos_sim += cosine(
                            current_weight.flatten(), layer_weights[cur_i + i][cur_j + j].flatten()
                        )
                        total += 1
        return dist / float(total), cos_sim / float(total)

    def calc_distances(self, layer_weights):
        # what is the height and width of the weight matrix
        dim = int(np.sqrt(layer_weights.shape[0]))
        # print('Weight dim:', dim, 'Creating shape:',(dim, dim)+layer_weights.shape[1:], layer_weights.shape)
        # reshape weights into square matrix
        layer_weights = layer_weights.reshape((dim, dim) + layer_weights.shape[1:])

        layer_weights = layer_weights / np.linalg.norm(layer_weights)

        for key in self.distance.keys():
            avg_dist, avg_cos_sim = self.avg_dist_window(layer_weights, window_size=int(key))
            self.distance[key].append(avg_dist)
            self.cosine[key].append(avg_cos_sim)

        return

    def on_train_begin(self, logs={}):
        w = self.model.layers[0].get_weights()[0]
        self.calc_distances(w)

        # self.print_res()
        return

    def print_res(self):
        for k in self.distance.keys():
            print(str(k) + ':', round(self.distance[k][-1], 4), end=' ')

        for k in self.cosine.keys():
            print('cos @',str(k) + ':', round(self.cosine[k][-1], 4), end=' ')
        print()

    def on_train_end(self, logs={}):
        # rename keys for euclidean distance results
        results = {'dist_%s' % k: self.distance[k] for k in self.distance.keys()}
        # rename keys for cosine similarity results
        results.update({'cos_%s' % k: self.cosine[k] for k in self.cosine.keys()})

        self.results = results

        return

    def on_epoch_end(self, epoch, logs={}):
        w = self.model.layers[0].get_weights()[0]
        self.calc_distances(w)

        # self.print_res()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
