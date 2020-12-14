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


# 加上DropOut
# 增加"视野"
class FcCvModelV3(Model):
    def __init__(self):
        super(FcCvModelV3, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout = layers.Dropout(0.2)
        # 1维卷积综合提取特征
        self.conv = Sequential([
            layers.Conv1D(filters=16,kernel_size=8,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv1D(filters=4, kernel_size=8, strides=8, padding='valid'),
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


# 加上DropOut
# 增加"视野"
# 更大网络
class FcCvModelV4(Model):
    def __init__(self):
        super(FcCvModelV4, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout = layers.Dropout(0.2)
        # 1维卷积综合提取特征
        self.conv = Sequential([
            layers.Conv1D(filters=16,kernel_size=8,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv1D(filters=4, kernel_size=8, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv1D(filters=2, kernel_size=8, strides=8, padding='valid'),
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


# 加上DropOut
# 增加"视野"
# 更大网络
# 增加dropout
class FcCvModelV5(Model):
    def __init__(self):
        super(FcCvModelV5, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 1.减少过拟合的情况
        # 2.模拟道数据的丢失
        self.dropout_fc = layers.Dropout(0.2)

        # 1维卷积综合提取特征
        self.conv1 = Sequential([
            layers.Conv1D(filters=16,kernel_size=9,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout_c1 = layers.Dropout(0.2)
        self.conv2 = Sequential([
            layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout_c2 = layers.Dropout(0.2)
        self.conv3 = Sequential([
            layers.Conv1D(filters=4, kernel_size=8, strides=8, padding='valid'),
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
        x = self.dropout_fc(x,training=training)
        x = self.conv1(x)
        x = self.dropout_c1(x)
        x = self.conv2(x)
        x = self.dropout_c2(x)
        x = self.conv3(x)
        x = tf.reshape(x,(batchSize,-1))
        x = self.fcThree(x)
        return x

# 加上DropOut
# 增加"视野"
# 更大网络
# 增加dropout
# 直接卷积
class FcCvModelV6(Model):
    def __init__(self):
        super(FcCvModelV6, self).__init__()

        # 大量的压缩数据
        self.conv = Sequential([
            layers.Conv1D(filters=32, kernel_size=256, strides=128, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        # 1.减少过拟合的情况
        # 2.模拟道数据的丢失
        self.dropout_c = layers.Dropout(0.2)

        # 1维卷积综合提取特征
        self.conv1 = Sequential([
            layers.Conv1D(filters=16,kernel_size=9,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c1 = layers.Dropout(0.2)
        self.conv2 = Sequential([
            layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        self.dropout_c2 = layers.Dropout(0.2)
        self.conv3 = Sequential([
            layers.Conv1D(filters=4, kernel_size=8, strides=8, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 最终作出异常体坐标与半径的预测
        self.fcThree = Sequential([
            layers.Dense(3)
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = self.conv(inputs)
        x = self.dropout_c(x,training=training)
        x = self.conv1(x)
        x = self.dropout_c1(x)
        x = self.conv2(x)
        x = self.dropout_c2(x)
        x = self.conv3(x)
        x = tf.reshape(x,(batchSize,-1))
        x = self.fcThree(x)
        return x


# V5基础上再次加大网络
# V5的训练集上loss7千多
# 加上DropOut
# 增加"视野"
# 增加卷积核
# 增加dropout
class FcCvModelV7(Model):
    def __init__(self):
        super(FcCvModelV7, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 1.减少过拟合的情况
        # 2.模拟道数据的丢失
        self.dropout_fc = layers.Dropout(0.3)

        # 1维卷积综合提取特征
        self.conv1 = Sequential([
            layers.Conv1D(filters=16,kernel_size=25,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout_c1 = layers.Dropout(0.3)
        self.conv2 = Sequential([
            layers.Conv1D(filters=8, kernel_size=25, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.dropout_c2 = layers.Dropout(0.3)
        self.conv3 = Sequential([
            layers.Conv1D(filters=4, kernel_size=8, strides=8, padding='valid'),
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
        x = self.dropout_fc(x,training=training)
        x = self.conv1(x)
        x = self.dropout_c1(x)
        x = self.conv2(x)
        x = self.dropout_c2(x)
        x = self.conv3(x)
        x = tf.reshape(x,(batchSize,-1))
        x = self.fcThree(x)
        return x


# 直接暴力全连接
# 网络单元数可能会变很大
class FcCvModelV8(Model):
    def __init__(self):
        super(FcCvModelV8, self).__init__()
        # 多层全链接的第一层，每一道数据的提取
        # 实际上是在每道上的滑动全连接
        self.fcOne = Sequential([
            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        # 1.减少过拟合的情况
        # 2.模拟道数据的丢失
        self.dropout_fc = layers.Dropout(0.3)

        # 第二、三层全连接
        # 得出预测
        self.fcTwo = Sequential([
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(3),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])

    def call(self, inputs, training=None, mask=None):
        batchSize, dao_num, data_num = inputs.shape
        x = tf.reshape(inputs, (-1, 1024))
        x = self.fcOne(x)
        x = tf.reshape(x, (batchSize,-1))
        x = self.dropout_fc(x,training=training)
        x = self.fcTwo(x)
        return x