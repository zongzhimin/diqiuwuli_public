import numpy as np
from Multi_FC_Model import MultiFCModel
from tensorflow.keras import optimizers
from multiFC_utils import data_shffle
import matplotlib.pyplot as plt
import tensorflow as tf

from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

model = MultiFCModel()
model.load_weights(r'E:\stuCode\diqiuwuli\pre_experiment2\weights\weights_s_30.w')


test_loss_all = 0
step_num = 0
for i in range(4):
    x_test = np.load('/openbayes/input/input1/x_test_512_'+str(i+1)+'.npy')
    y_test = np.load('/openbayes/input/input0/x_test_s_'+str(i+1)+'.npy')*1000
#         x_dev,y_dev = preprocess(x_dev, y_dev)
    y_test_pred = model(x_test)
    test_loss = tf.reduce_mean(tf.losses.MSE(y_test_pred, y_test))
    test_loss_all += test_loss
    step_num = step_num + 1
print("test_loss",(test_loss_all/step_num).numpy())


x_test = np.load('/openbayes/input/input1/x_test_512_'+str(1)+'.npy')
y_test = np.load('/openbayes/input/input0/x_test_s_'+str(1)+'.npy')
y_label = np.load('/openbayes/input/input0/x_test_'+str(1)+'.npy')

# 根据矩阵信息生成地层 512->512 双层
def data2layer(y_data):
    layer = np.zeros((512, 512), dtype=np.int32)
    for i in range(512):
        layer[int(y_data[0][i]):int(y_data[1][i]),i] = 1
        layer[int(y_data[1][i]):, i] = 2
    return layer

# 根据矩阵信息生成地层
def data2layer(y_data):
    layer = np.zeros((512, 512), dtype=np.int32)
    for i in range(512):
        layer[int(y_data[i]):,i] = 1
    return layer


j = np.random.randint(0, 512, size=(1))[0]
y_pred = model(x_test[j:j+1])/1000
y_t = y_test[j]
y_d = y_label[j]

# y_pred = model(x_test[j:j+1])
# y = y_test[j]

index = [i for i in range(512)]

plt.figure(figsize=(8,8))
plt.subplot(2, 1, 1)
plt.imshow(data2layer(y_d))
plt.subplot(2, 1, 2)
plt.plot(index,y_t,label='y_true')
plt.plot(index,y_pred[0],label='y_pred')
plt.legend()

