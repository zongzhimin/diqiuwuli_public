import numpy as np
import matplotlib.pyplot as plt
from data_process.readSu import read_su

# 读取su文件里的道数据
# 需要先安装 obspy
# conda config --add channels conda-forge
# conda install obspy
def read_su_example():
    traces = read_su('su file path')

# 示例
# 读取512个su文件的道数据到numpy数组
# 每个文件33道，每道10240个采样点
def su2numpy():
    # 数据放大最终在0-1之间 实际值需要调整
    amp = 1.9e6
    # 数据截取，开头的直达波部分的削弱，实际值需要调整
    epison = 5e-07

    # 所有数据
    data_all = np.zeros(512,33,10240)
    for file_i in range(2048):
        # 得到10240*33的数据
        traces = read_su('su_path' + str(file_i) + '.shot1')
        # 转置成33*10240
        traces = traces.T
        # 道数据直达波和反射波大小调整 过大的部分直接截取
        traces = traces.clip(-epison, epison)
        # 数据放大到 -1 到 1 之间
        traces = amp * traces
        data_all[file_i] = traces
    np.save('data_all',data_all)


