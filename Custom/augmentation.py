import numpy as np
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator


def swap_generator(x_train, y_train, batch_size=64):
    while True:
        for i in range(0, len(x_train), batch_size):
            # get batch of data
            x, y = x_train[i:i + batch_size], y_train[i:i + batch_size]
            tmp_x = deepcopy(x)
            hw = x.shape[1] / 2

            tmp_x[:, :hw, -hw:] = x[:, -hw:, :hw]
            tmp_x[:, -hw:, :hw] = x[:, :hw, -hw:]
            yield tmp_x, y
            del tmp_x


def noise_generator(x_train, y_train, noise_level=.15, batch_size=64):
    while True:
        for i in range(0, len(x_train), batch_size):
            # get batch of data
            x, y = x_train[i:i + batch_size], y_train[i:i + batch_size]
            # add random noise
            noise = np.random.normal(scale=noise_level, size=x.shape)

            yield x + noise, y
            del noise


def edge_noise_generator(x_train, y_train, noise_level=.05, batch_size=64):
    while True:
        for i in range(0, len(x_train), batch_size):
            # get batch of data
            x, y = x_train[i:i + batch_size], y_train[i:i + batch_size]
            # add random noise
            noise = np.random.normal(scale=noise_level, size=x.shape)
            noise[5:-5, 5:-5, :] = 0

            yield x + noise, y
            del noise


def get_batch_size(aug_amt, shape=(28, 28, 1)):
    x = np.zeros(shape=shape)
    dim = shape[1]
    batch_x = [];
    batch_y = []

    if aug_amt == 0:
        return 64
    else:
        aug_dim = int(aug_amt * dim)
        pad = np.pad(x.squeeze(), (aug_dim, aug_dim), 'constant', constant_values=0)
        for row in range(dim, pad.shape[0]):
            for col in range(dim, pad.shape[1]):
                batch_x.append(pad[row - dim:row, col - dim:col][:, :, np.newaxis])
                batch_y.append(y)

        return len(batch_x)


def translation(x_train, y_train, aug_amt=.5, batch_size=64, num_ex=int(60000 * .8)):
    dim = x_train.shape[1]
    batch_x = [];
    batch_y = []
    while True:
        for x, y in zip(x_train[:num_ex], y_train[:num_ex]):
            if aug_amt == 0:
                batch_x.append(x);
                batch_y.append(y)
                if len(batch_x) == batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = [];
                    batch_y = []
            else:
                aug_dim = int(aug_amt * dim)
                pad = np.pad(x.squeeze(), (aug_dim, aug_dim), 'constant', constant_values=0)
                for row in range(dim, pad.shape[0]):
                    for col in range(dim, pad.shape[1]):
                        batch_x.append(pad[row - dim:row, col - dim:col][:, :, np.newaxis])
                        batch_y.append(y)
                        if len(batch_x) == batch_size:
                            yield np.array(batch_x), np.array(batch_y)
                            batch_x = [];
                            batch_y = []


def get_generator(x_train, y_train, args):
    if args['aug_type'] == 'noise':
        datagen = noise_generator(x_train, y_train, args['aug'], batch_size=args['batch_size'])
    elif args['aug_type'] == 'edge_noise':
        datagen = edge_noise_generator(x_train, y_train, args['aug'], batch_size=args['batch_size'])
    else:
        # build generator for augmentation
        if args['aug_type'] == 'translation' or args['aug_type'] == 'vcp':

            datagen = ImageDataGenerator(
                width_shift_range=args['aug'],
                height_shift_range=args['aug'],
                fill_mode='nearest'
            )
        elif args['aug_type'] == 'rotation':
            datagen = ImageDataGenerator(
                rotation_range=int(args['aug']*360.)
            )

        datagen.fit(x_train)
        datagen = datagen.flow(x_train, y_train, batch_size=args['batch_size'])

    return datagen
