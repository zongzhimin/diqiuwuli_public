import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

"""
测试道数据的压缩与还原

"""


# 用全连接压缩与还原
class FCZipSeismicData(Model):
    def __init__(self):
        super(FCZipSeismicData, self).__init__()
        self.encoder = Sequential([
            layers.Dense(80),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.decoder = Sequential([
            layers.Dense(10240),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=None, mask=None):
        # batchSize, dao_num, data_num = inputs.shape
        # x = tf.reshape(inputs, (-1, 1024))
        x_en = self.encoder(inputs)
        x = self.decoder(x_en)
        # x = tf.reshape(x, (batchSize, dao_num, data_num))
        return x, x_en


# 用全连接压缩与还原
class FCZipSeismicDataV2(Model):
    def __init__(self):
        super(FCZipSeismicDataV2, self).__init__()
        self.encoder = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.decoder = Sequential([
            layers.Dense(1024),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x_en = self.encoder(x)
        x = self.decoder(x_en)
        x = tf.reshape(x, (batchSize, -1))
        return x, x_en


# 用全连接压缩与还原
class FCZipSeismicDataV3(Model):
    def __init__(self):
        super(FCZipSeismicDataV3, self).__init__()
        self.encoder = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.decoder = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1024)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x_en = self.encoder(x)
        x_hidden = tf.reshape(x_en, (batchSize, dao_num, 80))
        x = self.decoder(x_en)
        x = tf.reshape(x, (batchSize, dao_num, data_num))
        return x, x_en, x_hidden


class FCZipSeismicDataV4(Model):
    def __init__(self):
        super(FCZipSeismicDataV4, self).__init__()
        self.encoder = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(8),
            layers.BatchNormalization()
            # layers.LeakyReLU(alpha=0.2)
        ])
        self.decoder = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1024)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x_en = self.encoder(x)
        x_en = tf.nn.tanh(x_en)
        x_hidden = tf.reshape(x_en, (batchSize, dao_num, 80))
        x = self.decoder(x_en)
        x = tf.reshape(x, (batchSize, dao_num, data_num))
        return x, x_en, x_hidden
