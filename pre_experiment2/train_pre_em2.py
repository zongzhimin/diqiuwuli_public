import numpy as np
from Multi_FC_Model import MultiFCModel
from tensorflow.keras import optimizers
from multiFC_utils import data_shffle
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y*1000, dtype=tf.float32)
    return x, y


optimizer = optimizers.Adam(lr=0.01)
epochs = 30
model = MultiFCModel()

train_loss_list = []
dev_loss_list = []

for e in range(epochs):
    loss_train = 0
    step_num = 0
    for i in range(24):
        x = np.load('/openbayes/input/input1/x_train_512_' + str(i + 1) + '.npy')
        y = np.load('/openbayes/input/input0/x_train_s_' + str(i + 1) + '.npy') * 1000
        #         x,y = preprocess(x, y)
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.reduce_mean(tf.losses.MSE(y_pred, y))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_train += loss
        step_num = step_num + 1
    print("epochs:", e, " train_loss:", (loss_train / step_num).numpy())
    train_loss_list.append((loss_train / step_num).numpy())

    dev_loss_all = 0
    step_num = 0
    for i in range(4):
        x_dev = np.load('/openbayes/input/input1/x_dev_512_' + str(i + 1) + '.npy')
        y_dev = np.load('/openbayes/input/input0/x_dev_s_' + str(i + 1) + '.npy') * 1000
        #         x_dev,y_dev = preprocess(x_dev, y_dev)
        y_dev_pred = model(x_dev)
        dev_loss = tf.reduce_mean(tf.losses.MSE(y_dev_pred, y_dev))
        dev_loss_all += dev_loss
        step_num = step_num + 1
    dev_loss_list.append((dev_loss_all / step_num).numpy())
    print("epochs:", e, " dev_loss", (dev_loss_all / step_num).numpy())

model.save_weights('weights/weights_s_30.w')

np.save('loss/train_loss',train_loss_list)
np.save('loss/dev_loss',dev_loss_list)


test_loss_all = 0
step_num = 0
for i in range(4):
    x_test = np.load('/openbayes/input/input0/x_test_'+str(i+1)+'.npy')
    y_test = np.load('/openbayes/input/input0/y_test_512_'+str(i+1)+'.npy')
    x_test,y_test = preprocess(x_test, y_test)
    y_test_pred = model(x_test)
    test_loss = tf.reduce_mean(tf.losses.MSE(y_test_pred, y_test))
    test_loss_all += test_loss
    step_num = step_num + 1