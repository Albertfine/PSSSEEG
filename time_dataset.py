import numpy as np
import torch


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch


def batch_mask(dataset, batch_size, ratio):
    device = torch.device("cpu")
    dataset_size = len(dataset)
    if dataset_size > batch_size:
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
        batch = batch.to(device)
    else:
        n = int(batch_size/dataset_size) +1
        idx0 = torch.randperm(dataset_size*n)
        idx = torch.floor(idx0/n).int()
        batch_idx = idx[:batch_size]
        batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
        batch = batch.to(device)
    
    batch_num, data_length, data_dim = batch.shape
    len_keep = int(data_length*(1-ratio))
    noise = torch.rand(batch_num, data_length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    batch_sample = torch.gather(batch, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, data_dim))
    
    ids_mask = ids_shuffle[:, len_keep:]
    batch_mask = batch.scatter(1, ids_mask.unsqueeze(-1).repeat(1, 1, data_dim), 0)
    
    random_noise = torch.zeros(batch_size, data_length, data_dim)
    ids_shuffle_random = torch.zeros([batch_size, data_length, data_dim],dtype=torch.int64, device=device)
    for i in range(batch_size):
        random_noise[i,:,:] = torch.rand(data_length, data_dim, device=device)
        ids_shuffle_random[i,:,:] = torch.argsort(random_noise[i, :, :], dim=1)
    ids_mask_random = ids_shuffle_random[:, len_keep:, :]
    batch_mask_random = batch.scatter(1, ids_mask_random, 0)
    
    # mask = torch.ones([batch_size, data_length], device=device)
    # mask[:, :len_keep] = 0
    # mask = torch.gather(mask, dim=1, index=ids_restore)
    return batch, batch_sample, batch_mask, batch_mask_random
    



class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1) 
        data = data[::-1]  # reverse order

        norm_data = normalize(data)

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]  # array[seq_len, data.shape[1]]
            seq_data.append(x)  # list[len(norm_data) - seq_len + 1], every object is a array[seq_len, data.shape[1]]
        
        # shuffle
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class TimeDatasetEEG(torch.utils.data.Dataset):
    def __init__(self, dataEEG, seq_len):
        dataEEG = dataEEG.transpose((0, 2, 1))
        
        for i in range(len(dataEEG)):
            if i == 0:
                data = dataEEG[i]
            else:
                data = np.vstack([data, dataEEG[i]])
            
        data = data[::-1]  # reverse order

        norm_data = normalize(data)

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]  # array[seq_len, data.shape[1]]
            seq_data.append(x)  # list[len(norm_data) - seq_len + 1], every object is a array[seq_len, data.shape[1]]
        
        # shuffle
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
            
class TimeDatasetEEGorigin(torch.utils.data.Dataset):
    def __init__(self, dataEEG):
        data = dataEEG.transpose((0, 2, 1))

        seq_data = normalize(data)
        
        # shuffle
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
