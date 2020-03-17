import os
import argparse
import datetime
import subprocess
from tqdm import tqdm

print 'Current PID:',os.getpid()

parser = argparse.ArgumentParser()
parser.add_argument('--gpus',type=str, default='0,1,2,3',help='Available gpus separated by comma.')
parser.add_argument('--max_concurrent',type=int, default=24, help='Number of concurrent processes')

parser.add_argument('--num_layers',type=int, default=3)
parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'cifar'])
parser.add_argument('--save_type', type=str, default='none', choices=['none', 'json', 'all'])
FLAGS = parser.parse_args()

AUG_TYPES = ['noise', 'rotation', 'edge_noise']
SETTINGS  = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]

COMMAND = 'python main.py --dataset {dataset} --model {model} --aug {aug} --aug_type {aug_type} --vcp {vcp} --split_for_cifar 2'

stack = []

# # 0 augmentation
# for model in ['fcn', 'cnn']:
#     command = COMMAND.format(
#         dataset=FLAGS.dataset,
#         model=model,
#         aug=0.,
#         aug_type='translation',
#         vcp=0.,
#     )
#     stack.append(command)
#
# # translation augmentation
# for aug in SETTINGS:
#     for model in ['fcn', 'cnn']:
#         command = COMMAND.format(
#             dataset=FLAGS.dataset,
#             model=model,
#             aug=aug,
#             aug_type='translation',
#             vcp=0.,
#         )
#         stack.append(command)
#
# # variable connection patterns
# for aug in [0., .1, .2, .3, .4]:
#     for vcp in SETTINGS:
#         command = COMMAND.format(
#             dataset=FLAGS.dataset,
#             model='fcn',
#             aug=aug,
#             aug_type='vcp',
#             vcp=vcp,
#         )
#         stack.append(command)


# add other aug types at the end of queue
for aug_type in AUG_TYPES:
    for aug in SETTINGS:
        for model in ['fcn', 'cnn']:
            command = COMMAND.format(
                dataset=FLAGS.dataset,
                model=model,
                aug=aug,
                aug_type=aug_type,
                vcp=0.,
            )
            stack.append(command)


now = datetime.datetime.now()
print 'Starting @', now.strftime("%Y-%m-%d %H:%M")
print 'Total jobs to run:', len(stack)
gpus = { gpu:[] for gpu in FLAGS.gpus.split(',')}
num_per_gpu = FLAGS.max_concurrent / len(gpus)

pbar = tqdm(total=len(stack))

while stack:
    for gpu_id in gpus:
        while len(gpus[gpu_id]) < num_per_gpu:
            command = stack.pop(0).format(gpu=gpu_id)
            proc = subprocess.Popen([command + ' --gpu %s' % gpu_id],shell=True,stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)

            gpus[gpu_id].append(proc)

    GPUS_FULL = True
    while GPUS_FULL:
        for gpu_id in gpus:
            for i in range(len(gpus[gpu_id])-1,-1,-1):
                if gpus[gpu_id][i].poll() is not None:
                    gpus[gpu_id].pop(i)
                    GPUS_FULL = False
                    pbar.update(1)
                    break

pbar.close()