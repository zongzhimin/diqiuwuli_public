import numpy as np
import matplotlib.pyplot as plt


# 根据y_pred数据生成地层 1024*1024
# (0,15)(16,47)...(976,1007)(1008,1023)
# 宽:16,32...(中间共31个32)...32,16
def creatDiCeng(data_y):
    label = np.zeros((1024, 1024))
    label[int(data_y[0]):, 0:16] = 1
    label[int(data_y[32]):, 1008:] = 1
    for i in range(1, 32):
        label[int(data_y[i]):, 32 * (i - 1) + 16:32 * i + 16] = 1
    return label


# 根据y数据生成地层,平滑的，即中间点到中间点以直线连接，形成折线，然后填充 1024*1024
# (0,7)(8,31)(32,63)...(960-991)(992,1015)(1016,1023)
# 宽:8,24,32...(中间共30个32)...32.24.8
def creatDiCengSmooth(data_y):
    label = np.zeros((1024, 1024))
    # 边界宽8处
    label[int(data_y[0]):, 0:8] = 1
    label[int(data_y[32]):, 1016:] = 1
    # 边界宽24处
    for i, depth in enumerate(np.linspace(data_y[0], data_y[1], 24)):
        label[int(depth):, 8 + i] = 1
    for i, depth in enumerate(np.linspace(data_y[31], data_y[32], 24)):
        label[int(depth):, 992 + i] = 1
    # 中间宽32
    for i in range(1, 31):
        for j, depth in enumerate(np.linspace(data_y[i], data_y[i + 1], 32)):
            label[int(depth):, 32 * i + j] = 1
    return label


# 查看loss
def showLoss(dev_loss, train_loss):
    index = [i for i in range(len(dev_loss))]
    plt.title("dev loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.plot(index, dev_loss)
    plt.show()

    index = [i for i in range(len(train_loss))]
    plt.title("train loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.plot(index, train_loss)
    plt.show()

    if len(dev_loss) == len(train_loss):
        index = [i for i in range(len(dev_loss))]
        plt.title("loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.plot(index, dev_loss, c='red')
        plt.plot(index, train_loss, c='blue')
        plt.show()


# creatDiCengIndex中使用 ，交换位置
def y_exchange(y, i, j):
    e = y[0][i]
    y[0][i] = y[0][j]
    y[0][j] = e


# Model3随机产生地层坐标
def creatDiCengIndex():
    # 中间两点横坐标
    x = np.random.randint(0, 1024, size=(1, 2))
    # 4点的纵坐标
    y = np.random.randint(300, 600, size=(1, 4))
    x.sort()
    y.sort()

    # 随机产生向斜or背斜
    t = np.random.uniform(0, 1, size=(1, 4))
    t = np.where(t >= 0.5, 1, 0)

    if t[0][1] == 1:
        y_exchange(y, 0, 1)
    if t[0][2] == 1:
        y_exchange(y, 2, 3)

    if t[0][0] == 0:
        # 向斜
        y_exchange(y, 0, 1)
        y_exchange(y, 0, 2)
    else:
        # 背斜
        y_exchange(y, 1, 2)
        y_exchange(y, 2, 3)

    index = np.zeros((2, 4), dtype=int)

    index[0][1:3] = x
    index[1][0:4] = y
    index[0][3] = 1024
    print(index)
    print("-".join(str(i) for i in np.concatenate(index.T, axis=0)))


# 根据模型生成标签 512*512 -> 32
def model2Label():
    labels = np.zeros((841, 32), dtype=int)
    for k in range(1, 842):
        data = np.fromfile(r'E:\stuCode\segyFiles\Model4\512\model_512_' + str(k) + '.vp', dtype=np.float32)
        vpdata = data.reshape(512, 512)
        for i in range(32):
            for index, j in enumerate(vpdata.T[:, 16 * i + 8]):
                if j == 2000:
                    labels[k - 1, i] = index
                    break


# 读取su文件里道数据，调整数据并以矩阵存入np文件
def su2np():
    amp = 4e6
    epison = 2.5e-07
    x = np.zeros((841, 33, 10240), dtype=np.float32)
    for i in range(841):
        traces = read_su('/openbayes/input/input0/DENISE_MARMOUSI_y.su' + str(i + 1) + '.shot1')
        # 10240 * 33 转置成 33 * 10240
        traces = traces.T
        # 道数据直达波和反射波大小调整
        traces = traces.clip(-epison, epison)
        # 数据大约在 -1 到 1 之间
        traces = amp * traces
        x[i] = traces
    np.save('x', x)


def data_shffle(x, y):
    index = np.random.permutation(x.shape[0])
    return x[index, :, :], y[index, :]