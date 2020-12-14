import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential

"""
测试道数据的压缩与还原

"""

# 用全连接压缩与还原
class FCZipSeismicData(Model):
    def __init__(self):
        super(FCZipSeismicData, self).__init__()
        self.encoder = Sequential([
            layers.Dense(2),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.decoder = Sequential([
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('tanh')
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x = self.encoder(x)
        x = self.decoder(x)
        x = tf.reshape(inputs, (batchSize, dao_num, data_num))
        return x


# 用一维卷积压缩与还原
class ConvZipSeismicData(Model):
    def __init__(self):
        super(ConvZipSeismicData, self).__init__()
        self.encoder = Sequential([
            layers.Conv1D(filters=33, kernel_size=256, strides=128, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.decoder = Sequential([
            # layers.Con
        ])

    def call(self, inputs, training=None, mask=None):
        pass