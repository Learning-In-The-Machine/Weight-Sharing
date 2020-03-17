import os
import keras
import fcntl
import pprint
import argparse
import numpy as np
import tensorflow as tf

from model import Model
from keras.datasets import mnist, cifar10
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str)

parser.add_argument('--split_for_cifar',type=int, default=3)

parser.add_argument('--num_layers',type=int, default=3)
parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'cifar'])
parser.add_argument('--model',type=str, default='cnn', choices=['cnn', 'fcn'])

parser.add_argument('--vcp', type=float, default=0., help='Variable Connection Probability')
parser.add_argument('--aug', type=float, default=0., help='Translational Augmentation Probability')
parser.add_argument('--aug_type', default='translation', help='Type of Augmentation for training')
args = vars(parser.parse_args())

args['lr']         = 0.001 if args['dataset'] == 'mnist' else 0.0001
args['batch_size'] = 256   if args['dataset'] == 'mnist' else 128
args['patience']   = 50    if args['dataset'] == 'mnist' else 100
args['epochs']     = 250   if args['dataset'] == 'mnist' else 500

if args['split_for_cifar'] == 4:
    args['epochs'] = 1000

# SET GPU AND TF SETTINGS
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
config = tf.ConfigProto(); config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# LOAD DATA
if args['dataset'] == 'cifar':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3); x_test = x_test.reshape(-1, 32, 32, 3)
elif args['dataset'] == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1); x_test = x_test.reshape(-1, 28, 28, 1)

data = np.append(x_train,x_test,axis=0)
targets = keras.utils.to_categorical(np.append(y_train,y_test), 10)
data = data.astype('float64'); data /= 255.0

kf = KFold(n_splits=5); kf.get_n_splits(data)
METRICS = ['loss','val_loss','acc','val_acc','translation','noise','edge_noise','rotation','swap','dist_1','dist_4','cos_1','cos_4']

# 5 fold cross validation
for fold, (train_index, test_index) in enumerate(kf.split(data)):
    x_train, x_test = data[train_index], data[test_index]; y_train, y_test = targets[train_index], targets[test_index]

    model = Model(args)
    history = model.fit_generator(x_train, y_train, x_test, y_test)

    del model

    lines = ''
    for epoch in range(len(history['acc'])):
        lines += '{fold},{epoch},{model},{aug_type},{aug},{vcp},'.format(
            fold=fold,
            epoch=epoch+1,
            model=args['model'],
            aug_type=args['aug_type'],
            aug=args['aug'],
            vcp=args['vcp']
        )
        for metric in METRICS:
            if metric in history: lines += str(history[metric][epoch])
            lines += ','
        lines = lines[:-1] + '\n'
    if args['split_for_cifar'] == 4:
        results_file_name = 'Results/' + args['dataset'] + '_2.csv'
    elif args['split_for_cifar'] == 2:
        results_file_name = 'Results/'+args['dataset']+'_1.csv'
    else:
        results_file_name = 'Results/' + args['dataset'] + '.csv'

    with open(results_file_name, 'a+') as results_file:
        fcntl.flock(results_file, fcntl.LOCK_EX)
        if not results_file.readline():
            header = 'fold,epoch,model,aug_type,aug,vcp,'
            for metric in METRICS: header += metric + ','

            results_file.write(header[:-1] + '\n')

        results_file.write(lines)
        fcntl.flock(results_file, fcntl.LOCK_UN)