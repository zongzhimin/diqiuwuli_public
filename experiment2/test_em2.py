from experiment2.model_em2 import FcCvModel
import numpy as np
import matplotlib.pyplot as plt


# 1024*33*10240
data = np.load("")
labels = np.load()

test_data = data[896:]
test_y = data[896:]

model = FcCvModel()
model.load_weights('weights_1000')
y_test_pred = model(test_data)

num = 0

x,y,r = y_test_pred[num][0],y_test_pred[num][1],y_test_pred[num][2]
vp = np.ones((1024,1024),dtype=np.float32)
for i in range(x-r,x+r):
    for j in range(y-r,y+r):
        if (i-x)**2+(j-y)**2<r**2:
            vp[i,j] = 1

plt.figure(figsize=(18,12))
plt.subplot(1, 2, 1)
plt.title('预测位置')
plt.ylabel("深度")
plt.xlabel("水平")
plt.imshow(vp)

x,y,r = test_y[num][0],test_y[num][1],test_y[num][2]
vp = np.ones((1024,1024),dtype=np.float32)
for i in range(x-r,x+r):
    for j in range(y-r,y+r):
        if (i-x)**2+(j-y)**2<r**2:
            vp[i,j] = 1

plt.subplot(1, 2, 2)
plt.title('实际位置')
plt.ylabel("深度")
plt.xlabel("水平")
plt.imshow(vp)
plt.show()

