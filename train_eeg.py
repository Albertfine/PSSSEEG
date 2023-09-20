import os
import argparse
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm

torch.cuda.empty_cache()
torch.manual_seed(0)

from model_EEG import swin_eeg as create_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tmp = sio.loadmat('dataset.mat')
    # tmp = sio.loadmat(args.filename)
    # print(tmp['EEGsample'].shape)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    # print(xdata.shape)
    subIdx = np.array(tmp['subindex'])
    # 确保全部为整数
    label.astype(int)
    subIdx.astype(int)

    samplenum = xdata.shape[0]
    subnum = np.zeros([np.amax(subIdx), 3])

    for i in range(1, np.amax(subIdx) + 1):
        subnum[i - 1, 0] = subIdx[subIdx == i].size
        subnum[i - 1, 1] = sum(label[subIdx == i] == 0)
        subnum[i - 1, 2] = sum(label[subIdx == i] == 1)

    ydata = label

    trainindx = np.where(subIdx != 7)[0]
    x_train = xdata[trainindx]
    y_train = ydata[trainindx]

    testindx = np.where(subIdx == 7)[0]
    x_test = xdata[testindx]
    y_test = ydata[testindx]

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train),
                                                torch.from_numpy(y_train))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test),
                                               torch.from_numpy(y_test))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model = create_model(num_classes=args.num_classes).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    optimizer = optim.AdamW(pg, lr=args.lr)

    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loader = tqdm(train_loader)
        for n, data in enumerate(train_loader):
            inputs, labels = data
            input_data = inputs.to(device)
            class_label = labels.to(device)

            model.zero_grad()
            model.train()

            output = model(input_data)
            class_output = torch.max(output, dim=1)[1]
            loss = loss_func(class_output, class_label)

            loss.banckward()

            optimizer.step()
            optimizer.zero_grad()
        print('epoch is ', epoch)

        model.eval()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        val_loader = tqdm(val_loader)
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                test_inputs, test_labels = data
                sample_num += data.shape[0]

                test_input_data = test_inputs.to(device)
                tset_class_label = test_labels.to(device)

                pred = model(test_input_data)
                pred_classes = torch.max(pred, dim=1)[1]
                acc_num += torch.eq(pred_classes, tset_class_label).sum()

            print(acc_num / sample_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--filename', type=str, default=r'dataset.mat')

    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.0001)

    opt = parser.parse_args()
    # args = parser.parse_args()

    main(opt)
