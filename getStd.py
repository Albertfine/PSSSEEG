import numpy as np


def datastd(data):
    datamax = np.amax(data)
    datamin = np.amin(data)
    data = (data + abs(datamin)) / (abs(datamax) + abs(datamin) + 1e-7)

    [N, W, H] = data.shape
    data = data.reshape(N, 1, -1)  # [B, NC, N] -> [B, 1, NC*N]

    data_mean = data.mean(2).sum(0)
    data_std = data.std(2).sum(0)
    data_num = N
    data_mean = data_mean / data_num
    data_std = data_std / data_num
    data = data.reshape(N, W, H)
    for i in range(N):
        data[i, :, :] = (data[i, :, :] - data_mean) / data_std

    return data


def datanorm(data):
    datamax = np.amax(data)
    datamin = np.amin(data)
    data = (data + abs(datamin)) / (abs(datamax) + abs(datamin))

    return data
