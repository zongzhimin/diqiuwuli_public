import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential,optimizers
import numpy as np
from experiment1.model_em1 import FCZipSeismicDataV3


x = np.load(r'E:\dataaaaaaaaaa\x.npy')
inputs = x[0:32]

lr = 5e-4
epochs = 1000
L1_penalty = 1
optimizer = optimizers.Adam(learning_rate=lr)

model = FCZipSeismicDataV3()

loss_log_path = './tf_dir/loss_fc_encoder_decoder9'
loss_summary_writer = tf.summary.create_file_writer(loss_log_path)

for epoch in range(20000):
    with tf.GradientTape() as tape:
        x_r,x_en,x_hidden = model(inputs)
        l1_loss = L1_penalty * tf.reduce_mean(tf.abs(x_r - inputs))
    grads = tape.gradient(l1_loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    with loss_summary_writer.as_default():
        tf.summary.scalar('train_loss', l1_loss, step=epoch)
    if (epoch+1)%500 == 0:
        model.save_weights('weights2/weights_'+str(epoch+1))