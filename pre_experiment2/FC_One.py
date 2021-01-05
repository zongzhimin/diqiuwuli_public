import tensorflow as tf
from tensorflow.keras import Model, layers


class FcOne(Model):
    def __init__(self):
        super(FcOne, self).__init__()
        #         self.fc1 = layers.Dense(512)
        #         self.bn1 = layers.BatchNormalization()
        #         self.ac1 = layers.Activation('relu')

        self.fc2 = layers.Dense(128)
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.Activation('relu')

        self.fc3 = layers.Dense(64)
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.Activation('relu')

    def call(self, inputs, training=None, mask=None):
        #         x = self.fc1(inputs)
        #         x = self.bn1(x)
        #         x = self.ac1(x)

        x = self.fc2(inputs)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.ac3(x)

        return x
