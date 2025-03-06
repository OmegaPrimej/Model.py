import tensorflow as tf
from tensorflow import keras
from keras.layers import *

class HollywoodVideoGen(keras.Model):
  def __init__(self):
    super(HollywoodVideoGen, self).__init__()
    self.encoder = keras.layers.LSTM(128)
    self.decoder = keras.layers.LSTM(128)
    self.dense = keras.layers.Dense(256)

  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    output = self.dense(decoded)
    return output
