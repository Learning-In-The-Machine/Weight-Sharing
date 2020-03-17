import sherpa
import argparse
import datetime
import itertools

import sys
sys.path.append('../')
from utils import build_directory

parser = argparse.ArgumentParser()
parser.add_argument('--gpus',type=str, default='0,1,2,3',help='Available gpus separated by comma.')
parser.add_argument('--max_concurrent',type=int, default=4, help='Number of concurrent processes')

parser.add_argument('--max_layers',type=int, default=3)
parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'cifar'])
parser.add_argument('--save_type', type=str, default='json', choices=['none', 'json', 'all'])
FLAGS = parser.parse_args()

parameters = [
    sherpa.Choice('batch_size', [256 if FLAGS.dataset == 'mnist' else 128]),
    sherpa.Choice('patience', [25 if FLAGS.dataset == 'mnist' else 50]),
    sherpa.Choice('epochs', [100 if FLAGS.dataset == 'mnist' else 200]),
    sherpa.Choice('model', ['fcn', 'cnn']),
    sherpa.Choice('num_layers', [2, FLAGS.max_layers]),
    sherpa.Choice('lr', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
    # sherpa.Choice('activation', ['relu', 'prelu', 'elu', 'leaky_relu', 'sigmoid']),
    # sherpa.Choice('kernel_initializer', ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
]

for k,v in vars(FLAGS).iteritems():
    parameters.append(
        sherpa.Choice(k, [v])
    )

# Run on local machine.
gpus = [int(x) for x in FLAGS.gpus.split(',')]
processes_per_gpu = FLAGS.max_concurrent//len(gpus)
assert FLAGS.max_concurrent%len(gpus) == 0
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))

sched = sherpa.schedulers.LocalScheduler(resources=resources)
alg = sherpa.algorithms.RandomSearch(max_num_trials=200)

build_directory('SherpaResults/' + FLAGS.dataset + '/Models')

sherpa.optimize(
    parameters=parameters,
    algorithm=alg,
    lower_is_better=True,
    command='python hp_main.py',
    scheduler=sched,
    max_concurrent=FLAGS.max_concurrent,
    output_dir='SherpaResults/' + FLAGS.dataset +'/'
)
