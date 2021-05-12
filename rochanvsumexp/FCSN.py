import pprint
from torch.utils.data import DataLoader
import h5py
import torch
import numpy as np
from knapsack import knapsack
import torch.nn as nn
from collections import OrderedDict

####################################################################
# -------------------------- CONFIG CLASS --------------------------
####################################################################

class Config():
    """Config class"""
    def __init__(self, **kwargs):
        # Path
        self.data_path = './data/fcsn_tvsum.h5'
        self.save_dir = 'save_dir'
        self.score_dir = 'score_dir'
        self.log_dir = 'log_dir'

        # Model
        self.mode = 'train'
        self.gpu = True
        self.n_epochs = 50
        self.n_class = 2
        self.lr = 1e-3
        self.momentum = 0.9
        self.batch_size = 5

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)
        return config_str


if __name__ == '__main__':
    config = Config()
    print(config)

####################################################################
# -------------------------- DATA LOADER CLASS ---------------------
####################################################################


class VideoData(object):
    """Dataset class"""

    def __init__(self, data_path):
        self.data_file = h5py.File(data_path)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        index += 1
        video = self.data_file['video_' + str(index)]
        feature = torch.tensor(video['feature'][()]).t()
        label = torch.tensor(video['label'][()], dtype=torch.long)
        return feature, label, index


def get_loader(path, batch_size=5):
    dataset = VideoData(path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [len(dataset) - len(dataset) // 5, len(dataset) // 5])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, test_dataset


if __name__ == '__main__':
    loader = get_loader('fcsn_dataset.h5')


####################################################################
# ----------------------- EVALUATION METHODS ----------------------
####################################################################


def eval_metrics(y_pred, y_true):
    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return [precision, recall, fscore]


def select_keyshots(video_info, pred_score):
    N = video_info['length'][()]
    cps = video_info['change_points'][()]
    weight = video_info['n_frame_per_seg'][()]
    pred_score = np.array(pred_score.cpu().data)
    pred_score = upsample(pred_score, N)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    _, selected = knapsack(pred_value, weight, int(0.15 * N))
    selected = selected[::-1]
    key_labels = np.zeros(shape=(N, ))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1
    return pred_score.tolist(), selected, key_labels.tolist()


def upsample(down_arr, N):
    up_arr = np.zeros(N)
    ratio = N // 320
    l = (N - ratio * 320) // 2
    i = 0
    while i < 320:
        up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i]
        l += ratio
        i += 1
    return up_arr

####################################################################
# ----------------------- EVALUATION METHODS ----------------------
####################################################################

class FCSN(nn.Module):
    def __init__(self, n_class=2):
        super(FCSN, self).__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_1', nn.BatchNorm1d(1024)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_2', nn.BatchNorm1d(1024)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(2, stride=2, ceil_mode=True))
        ]))  # 1/2

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_1', nn.BatchNorm1d(1024)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_2', nn.BatchNorm1d(1024)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool1d(2, stride=2, ceil_mode=True))
        ]))  # 1/4

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_1', nn.BatchNorm1d(1024)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_2', nn.BatchNorm1d(1024)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_3', nn.BatchNorm1d(1024)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool1d(2, stride=2, ceil_mode=True))
        ]))  # 1/8

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv1d(1024, 2048, 3, padding=1)),
            ('bn4_1', nn.BatchNorm1d(2048)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_2', nn.BatchNorm1d(2048)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_3', nn.BatchNorm1d(2048)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool1d(2, stride=2, ceil_mode=True))
        ]))  # 1/16

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_1', nn.BatchNorm1d(2048)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_2', nn.BatchNorm1d(2048)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_3', nn.BatchNorm1d(2048)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool1d(2, stride=2, ceil_mode=True))
        ]))  # 1/32

        self.conv6 = nn.Sequential(OrderedDict([
            ('fc6', nn.Conv1d(2048, 4096, 1)),
            ('bn6', nn.BatchNorm1d(4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('drop6', nn.Dropout())
        ]))

        self.conv7 = nn.Sequential(OrderedDict([
            ('fc7', nn.Conv1d(4096, 4096, 1)),
            ('bn7', nn.BatchNorm1d(4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout())
        ]))

        self.conv8 = nn.Sequential(OrderedDict([
            ('fc8', nn.Conv1d(4096, n_class, 1)),
            ('bn8', nn.BatchNorm1d(n_class)),
            ('relu8', nn.ReLU(inplace=True)),
        ]))

        self.conv_pool4 = nn.Conv1d(2048, n_class, 1)
        self.bn_pool4 = nn.BatchNorm1d(n_class)

        self.deconv1 = nn.ConvTranspose1d(n_class, n_class, 4, padding=1, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose1d(n_class, n_class, 16, stride=16, bias=False)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        pool4 = h

        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)

        h = self.deconv1(h)
        upscore2 = h

        h = self.conv_pool4(pool4)
        h = self.bn_pool4(h)
        score_pool4 = h

        h = upscore2 + score_pool4

        h = self.deconv2(h)

        return h


if __name__ == '__main__':
    import torch

    net = FCSN()
    data = torch.randn((1, 1024, 320))
    print(net(data).shape)

####################################################################
# ----------------------- EVALUATION METHODS ----------------------
####################################################################