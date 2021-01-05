from experiment2.model_em2 import FcCvModel
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np

lr = 5e-4
epochs = 1000

optimizer = optimizers.Adam(learning_rate=lr)

# 1024*33*10240
data = np.load("")
labels = np.load()

train_data = data[0:768]
train_y = labels[0:768]
dev_data = data[768:896]
dev_y = labels[768:896]
test_data = data[896:]
test_y = data[896:]

model = FcCvModel()

# train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
# dev_loss = tf.keras.metrics.Mean('dev_loss', dtype=tf.float32)

loss_log_path = './tf_dir/loss_all'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)


for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(train_data)
        loss = tf.losses.MSE(train_y,y_pred)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    # train_loss(loss)
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=epoch)

    if epochs%100 == 0:
        y_dev_pred = model(dev_data)
        dev_loss = tf.losses.MSE(dev_y,y_dev_pred)
        with loss_summary_writer.as_default():
            tf.summary.scalar('dev_loss', dev_loss, step=epoch)

y_test_pred = model(test_data)
test_loss = tf.losses.MSE(test_y,y_test_pred)
print("test_loss:",test_loss.numpy())
