import tensorflow as tf
from tensorflow.keras import optimizers
from pre_experiment1.encodeDecode1024 import EncodeDecode1024
import numpy as np


x_train = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\x_train.npy')
y_train = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\y_train.npy')
x_dev = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\x_dev.npy')
y_dev = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\y_dev.npy')
x_test = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\x_test.npy')
y_test = np.load(r'E:\stuCode\diqiuData\npData\data1024\data3\y_test.npy')

#
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dev_db = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess(x, y):
    x = tf.reshape(tf.cast(x, dtype=tf.float32), shape=(1024, 32, 1))
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y,axis=-1,depth=2)
    return x, y


bactchsz = 2
optimizer = optimizers.Adam(lr=0.001)
epochs = 1000

file_train_w=open(r'E:\stuCode\diqiuData\npData\data1024\data3\loss\train.txt',mode='a')
file_dev_w=open(r'E:\stuCode\diqiuData\npData\data1024\data3\loss\dev.txt',mode='a')

train_db = train_db.map(preprocess).shuffle(1000).batch(bactchsz)
dev_db = dev_db.map(preprocess).shuffle(1000).batch(bactchsz)
test_db = test_db.map(preprocess).shuffle(1000).batch(bactchsz)

model = EncodeDecode1024()

model.build(input_shape=(None, 1024, 32, 1))
list_train_loss = []
list_dev_loss = []

for i in range(epochs):
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.losses.categorical_crossentropy(y,y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

    print("epoch:",i," train loss:",loss)
    dev_loss = 0
    dev_num = 0
    for (x,y) in dev_db:
        y_pred = model(x)
        loss = tf.losses.categorical_crossentropy(y, y_pred)
        loss = tf.reduce_mean(loss)
        dev_loss += loss
        dev_num += 1
    dev_loss = dev_loss/dev_num
    print("epoch:", i, " dev loss:", dev_loss)
    list_train_loss.append(loss)
    list_dev_loss.append(dev_loss)
    if i % 100 == 0:
        model.save_weights(r"E:\stuCode\diqiuData\npData\data1024\data3\weights\modelWeights-" + str(i))
        file_train_w.write(" ".join(str(i) for i in list_train_loss)+" ")
        file_dev_w.write(" ".join(str(i) for i in list_dev_loss)+" ")
        file_train_w.flush()
        file_dev_w.flush()
        list_train_loss = []
        list_dev_loss = []

test_loss = 0
test_num = 0
for (x,y) in test_db:
    y_pred = model(x)
    loss = tf.losses.categorical_crossentropy(y, y_pred)
    loss = tf.reduce_mean(loss)
    test_loss += loss
    test_num += 1
test_loss = test_loss/test_num
print("epoch:", i, " test loss:", test_loss)
