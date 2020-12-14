import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential,optimizers
import numpy as np
from experiment1.model_em1 import FCZipSeismicData


x = np.load(r'E:\dataaaaaaaaaa\x.npy')

lr = 5e-4
epochs = 1000
L1_penalty = 1
optimizer = optimizers.Adam(learning_rate=lr)

model = FCZipSeismicData()

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        x_r = model(x)
        l1_loss = L1_penalty * tf.reduce_mean(tf.abs(x_r - x))
    grads = tape.gradient(l1_loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    print(l1_loss)