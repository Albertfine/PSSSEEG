import numpy as np


def datastd1D(data):

    [N, C, S] = data.shape
    datamax = np.zeros(C)
    datamin = np.zeros(C)
    for i in range(C):
        datamax[i] = np.amax(data[:, i, :])
        datamin[i] = np.amin(data[:, i, :])
        data[:, i, :] = (data[:, i, :] + abs(datamin[i])) / (abs(datamax[i]) + abs(datamin[i]) + 1e-7)

    data = data.reshape(N, C, -1)  # [B, C, 1, N] -> [B, C, 1*N]

    data_mean = data.mean(2).sum(0)
    data_std = data.std(2).sum(0)
    data_num = N
    data_mean = data_mean / data_num
    data_std = data_std / data_num

    for i in range(C):
        data[:, i, :] = (data[:, i, :] - data_mean[i]) / data_std[i]
    data = data.reshape(N, C, 1, -1)

    return data


def datanorm(data):
    datamax = np.amax(data)
    datamin = np.amin(data)
    data = (data + abs(datamin)) / (abs(datamax) + abs(datamin))

    return data
