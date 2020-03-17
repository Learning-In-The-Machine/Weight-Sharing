import json
import keras
import numpy as np
import keras.backend as K
from augmentation import noise_generator, edge_noise_generator, swap_generator
from keras.preprocessing.image import ImageDataGenerator


class ValidationGeneration(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, args):
        super(ValidationGeneration, self).__init__()
        self.data = x_test
        self.label = y_test
        self.args = args

        self.metrics = {
            'translation': [],
            'noise': [],
            'edge_noise': [],
            'rotation': [],
            'swap': [],
        }
        # create generator to translate data
        self.translation = ImageDataGenerator(
            width_shift_range=.25,
            height_shift_range=.25,
            fill_mode='nearest'
        );
        self.translation.fit(self.data)

        self.rotation = ImageDataGenerator(
            rotation_range=45
        );
        self.rotation.fit(self.data)

    def on_epoch_end(self, epoch, logs={}):
        # translation
        results = self.model.evaluate_generator(
            self.translation.flow(self.data, self.label, batch_size=self.args['batch_size']),
            len(self.data) // self.args['batch_size']
        );
        self.metrics['translation'].append(results[1])

        # noise
        results = self.model.evaluate_generator(
            noise_generator(
                self.data, self.label,
                noise_level=.15,
                batch_size=self.args['batch_size']
            ),
            len(self.data) // self.args['batch_size']
        );
        self.metrics['noise'].append(results[1])

        # rotation
        results = self.model.evaluate_generator(
            self.rotation.flow(self.data, self.label,batch_size=self.args['batch_size']),
            len(self.data) // self.args['batch_size']
        );
        self.metrics['rotation'].append(results[1])

        # noise
        results = self.model.evaluate_generator(
            edge_noise_generator(
                self.data, self.label,
                noise_level=.15,
                batch_size=self.args['batch_size']
            ),
            len(self.data) // self.args['batch_size']
        );
        self.metrics['edge_noise'].append(results[1])

        # swap
        results = self.model.evaluate_generator(
            swap_generator(
                self.data, self.label,
                batch_size=self.args['batch_size']
            ),
            len(self.data) // self.args['batch_size']
        );
        self.metrics['swap'].append(results[1])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return
