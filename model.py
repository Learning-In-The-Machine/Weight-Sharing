from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.layers.local import LocallyConnected2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

from Custom.layers import VariableConnections
from Custom.augmentation import get_generator
from Custom.validation_generation import ValidationGeneration
from Custom.measure_distance import MeasureDistance

class Model:
    def __init__(self, args, file_name=None):
        self.args = args

        if file_name is not None:
            self._load_model(file_name)
        else:
            self._build_model()

    def fit(self, x_train, y_train, x_test, y_test, callbacks=[]):
        callbacks.append(
            EarlyStopping(monitor='val_acc', min_delta=0, patience=self.args['patience'], verbose=0, mode='auto')
        )

        # training
        history = self.model.fit(
            x_train, y_train,
            epochs=self.args['epochs'],
            validation_data=(x_test, y_test),
            batch_size=self.args['batch_size'],
            callbacks=callbacks,
            verbose=2
        )

        return history.history

    def fit_generator(self, x_train, y_train, x_test, y_test, callbacks=[]):
        callbacks.append(
            EarlyStopping(monitor='val_acc', min_delta=0, patience=self.args['patience'], verbose=0, mode='auto')
        )

        val_gen = ValidationGeneration(x_test,y_test,self.args); callbacks.append(val_gen)

        if self.args['model'] == 'fcn' and self.args['vcp'] == 0:
            measure_dist = MeasureDistance()
            callbacks.append(measure_dist)

        # training
        history = self.model.fit_generator(
            get_generator(x_train, y_train, self.args),
            steps_per_epoch=x_train.shape[0] // self.args['batch_size'],
            epochs=self.args['epochs'],
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            # verbose=0
        )
        # store results from augmenting the validation set
        history.history.update(val_gen.metrics)

        # add filter distance measurement results
        if self.args['model'] == 'fcn' and self.args['vcp'] == 0:
            for k in measure_dist.results.keys(): measure_dist.results[k] = measure_dist.results[k][:-1]
            history.history.update(measure_dist.results)

        return history.history

    def _build_model(self):
        if self.args['dataset'] == 'mnist':
            model = self._mnist_net()
        else:
            model = self._cifar_net()

        self.model = self._compile(model)

    def _compile(self, model):
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=self.args['lr']),
            metrics=['accuracy']
        )
        return model

    def _mnist_net(self):
        layer = LocallyConnected2D if self.args['model'] == 'fcn' else Conv2D
        model = Sequential()
        if self.args['model'] == 'fcn' and self.args['vcp'] > 0: model.add(VariableConnections(self.args['vcp']))
        model.add(layer(32, (3, 3), strides=2, activation='relu', padding='valid', input_shape=(28, 28, 1)))
        if self.args['model'] == 'fcn' and self.args['vcp'] > 0: model.add(VariableConnections(self.args['vcp']))
        model.add(layer(64, (3, 3), strides=2, activation='relu', padding='valid'))

        if self.args['num_layers'] == 3:
            if self.args['model'] == 'fcn' and self.args['vcp'] > 0: model.add(VariableConnections(self.args['vcp']))
            model.add(layer(128, (3, 3), activation='relu', padding='valid'))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def _mnist_net(self):
        layer = LocallyConnected2D if self.args['model'] == 'fcn' else Conv2D
        model = Sequential()
        if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
            model.add(VariableConnections(self.args['vcp'], input_shape=(28,28,1)))
            model.add(layer(32, (3,3), strides=2, activation='relu', padding='valid'))
        else:
            model.add(layer(32, (3,3), strides=2, activation='relu', padding='valid', input_shape=(28,28,1)))

        if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
            model.add(VariableConnections(self.args['vcp']))

        model.add(layer(64, (3,3), strides=2, activation='relu', padding='valid'))
        if self.args['num_layers'] == 3:
            if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
                model.add(VariableConnections(self.args['vcp']))
            model.add(layer(128, (3, 3), activation='relu', padding='valid'))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def _cifar_net(self):
        layer = LocallyConnected2D if self.args['model'] == 'fcn' else Conv2D
        model = Sequential()
        if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
            model.add(VariableConnections(self.args['vcp'], input_shape=(32, 32, 3)))
            model.add(layer(64, (5, 5), strides=2, activation='relu', padding='valid'))
        else:
            model.add(layer(64, (5,5), strides=2, activation='relu', padding='valid', input_shape=(32, 32, 3)))

        if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
            model.add(VariableConnections(self.args['vcp']))

        model.add(layer(128, (5,5), strides=2, activation='relu', padding='valid'))
        if self.args['num_layers'] == 3:
            if self.args['model'] == 'fcn' and self.args['vcp'] > 0:
                model.add(VariableConnections(self.args['vcp']))
            model.add(layer(256, (3, 3), activation='relu', padding='valid'))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def _load_model(self, file_name, load_type='json'):
        if load_type == 'json':
            with open('model.json', 'r') as json_file:
                loaded_model = model_from_json(json_file.read())
        elif load_type == 'all':
            self.model = load_model(file_name)

    def save_model(self, file_name):
        if self.args['save_type'] == 'json':
            model_json = self.model.to_json()
            with open(file_name+'.json', 'w') as json_file:
                json_file.write(model_json)

        elif self.args['save_type'] == 'all':
            self.model.save(file_name+'.h5')

