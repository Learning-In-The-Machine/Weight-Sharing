import keras
import numpy as np
from keras.layers import Layer
from keras import backend as K

class VariableConnections(Layer):
    def __init__(self,drop_prob=0.25, **kwargs):
        self.drop_prob = drop_prob
        super(VariableConnections, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        x = np.random.choice([1,0], size=input_shape[1:], p=[1-self.drop_prob, self.drop_prob])
        self.kernel = K.cast(x,dtype='float32')
        super(VariableConnections, self).build(input_shape)

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape