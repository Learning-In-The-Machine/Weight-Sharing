import os
import keras
import pprint
import argparse
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from model import Model
from keras.datasets import mnist, cifar10

parser = argparse.ArgumentParser()
parser.add_argument("--notsherpa", default=False, action='store_true')
parser.add_argument('--gpu', type=str, default='')
args =  parser.parse_args()

tf.set_random_seed(0); np.random.seed(0)

gpu = os.environ.get("SHERPA_RESOURCE", '')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu or args.gpu)

config = tf.ConfigProto(); config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

##############################
import sherpa
client = sherpa.Client()
trial = client.get_trial()

pp = pprint.PrettyPrinter(indent=4); pp.pprint(trial.parameters)
##############################

if trial.parameters['dataset'] == 'cifar':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3); x_test = x_test.reshape(-1, 32, 32, 3)
elif trial.parameters['dataset'] == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1); x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype('float64'); x_test = x_test.astype('float64')
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)
x_train /= 255.; x_test /= 255.

model = Model(trial.parameters)
# sherpa_callback = client.keras_send_metrics(trial, objective_name='val_acc',context_names=['acc', 'val_acc', 'loss', 'val_loss'])

history = model.fit(x_train, y_train, x_test, y_test)

# send metrics to sherpa
for epoch in range(len(history['acc'])):
    context = {
        'acc': history['acc'][epoch],
        'loss': history['loss'][epoch],
        'val_acc': history['val_acc'][epoch],
        'val_loss': history['val_loss'][epoch],
    }
    client.send_metrics(trial, epoch+1, context['val_loss'], context)

model.save_model(
    'SherpaResults/{dataset}/Models/{id}.h5'.format(
        dataset=trial.parameters['dataset'],
        id='%05d' % trial.id
    )
)
