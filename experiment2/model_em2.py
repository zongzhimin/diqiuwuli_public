import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model


class FcCvModel(Model):
    def __init__(self):
        super(FcCvModel, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(2),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 1维卷积综合提取特征
        self.conv = Sequential([
            layers.Conv1D(filters=4,kernel_size=5,strides=5),

            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 最终作出异常体坐标与半径的预测
        self.fcThree = Sequential([
            layers.Dense(3)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x = self.fcOne(x)
        x = tf.reshape(x, (batchSize, dao_num,-1))
        x = tf.transpose(x,perm=[0,2,1])
        x = self.conv(x)
        x = tf.reshape(x,(batchSize,-1))
        x = self.fcThree(x)
        return x


# 加上DropOut
class FcCvModelV2(Model):
    def __init__(self):
        super(FcCvModelV2, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(2),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout = layers.Dropout(0.2)
        # 1维卷积综合提取特征
        self.conv = Sequential([
            layers.Conv1D(filters=4,kernel_size=5,strides=5),

            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 最终作出异常体坐标与半径的预测
        self.fcThree = Sequential([
            layers.Dense(3)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x = self.fcOne(x)
        x = tf.reshape(x, (batchSize, dao_num,-1))
        x = tf.transpose(x,perm=[0,2,1])
        x = self.dropout(x,training=training)
        x = self.conv(x)
        x = tf.reshape(x,(batchSize,-1))
        x = self.fcThree(x)
        return x

