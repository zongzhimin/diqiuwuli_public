import numpy as np
from model_em3 import FcCvModelV
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf


optimizer = optimizers.Adam(lr=0.01)
epochs = 300
model = FcCvModelV()

loss_log_path = './tf_dir/loss_all_Fcov_xiangxiebeixie'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)

train_loss_m = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
dev_loss_m = tf.keras.metrics.Mean('dev_loss', dtype=tf.float32)

for e in range(epochs):
    for i in range(24):
        x = np.load('/openbayes/input/input1/x_train_512_' + str(i + 1) + '.npy')
        y = np.load('/openbayes/input/input0/x_train_s_' + str(i + 1) + '.npy') * 1000
        for step in range(2):
            with tf.GradientTape() as tape:
                y_pred = model(x[step*256:(step+1)*256])
                loss = tf.reduce_mean(tf.losses.MSE(y_pred, y[step*256:(step+1)*256]))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_m(loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss_m.result(), step=e)
    train_loss_m.reset_states()

    for i in range(4):
        x_dev = np.load('/openbayes/input/input1/x_dev_512_' + str(i + 1) + '.npy')
        y_dev = np.load('/openbayes/input/input0/x_dev_s_' + str(i + 1) + '.npy') * 1000
        for step in range(2):
            y_dev_pred = model(x_dev[step*256:(step+1)*256])
            dev_loss = tf.reduce_mean(tf.losses.MSE(y_dev_pred, y_dev[step*256:(step+1)*256]))
            dev_loss_m(dev_loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('dev_loss', dev_loss_m.result(), step=e)
    dev_loss_m.reset_states()

    model.save_weights('weights/weights_' + str(e + 1))




