import tensorflow as tf
from tensorflow.keras import Model, layers
from pre_experiment2.FC_One import  FcOne
from pre_experiment2.FC_Two import FCTwo


class MultiFCModel(Model):
    def __init__(self):
        super(MultiFCModel, self).__init__()
        self.fc_One = FcOne()
        self.fc_two = FCTwo()

    def call(self, inputs, training=None, mask=None):
        # m*33*1024
        m, dao_num, data_num = inputs.shape
        # =>(m*33)*1024
        x = tf.reshape(inputs, shape=(-1, data_num))
        # =>(m*33)*2
        x = self.fc_One(x)
        # =>(m*66)
        x = tf.reshape(x, shape=(m, -1))
        # =>(m*33)
        x = self.fc_two(x)
        return x
