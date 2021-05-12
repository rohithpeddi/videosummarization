import pprint
from torch.utils.data import DataLoader
import h5py
import torch
import torch.nn as nn
from collections import OrderedDict

import json
import csv
import h5py
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import pdb

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
# ----------------------- KNAPSACK METHODS ------------------------
####################################################################

# This file was found online, but I am sorry that I don’t know who the original author is now.
# http://www.geeksforgeeks.org/knapsack-problem/


def knapsack(v, w, max_weight):
    rows = len(v) + 1
    cols = max_weight + 1

    # adding dummy values as later on we consider these values as indexed from 1 for convinence

    v = np.r_[[0], v]
    w = np.r_[[0], w]

    # row : values , #col : weights
    dp_array = [[0 for i in range(cols)] for j in range(rows)]

    # 0th row and 0th column have value 0

    # values
    for i in range(1, rows):
        # weights
        for j in range(1, cols):
            # if this weight exceeds max_weight at that point
            if j - w[i] < 0:
                dp_array[i][j] = dp_array[i - 1][j]

            # max of -> last ele taken | this ele taken + max of previous values possible
            else:
                dp_array[i][j] = max(dp_array[i - 1][j], v[i] + dp_array[i - 1][j - w[i]])

    # return dp_array[rows][cols]  : will have the max value possible for given wieghts

    chosen = []
    i = rows - 1
    j = cols - 1

    # Get the items to be picked
    while i > 0 and j > 0:

        # ith element is added
        if dp_array[i][j] != dp_array[i - 1][j]:
            # add the value
            chosen.append(i - 1)
            # decrease the weight possible (j)
            j = j - w[i]
            # go to previous row
            i = i - 1

        else:
            i = i - 1

    return dp_array[rows - 1][cols - 1], chosen


# main
if __name__ == "__main__":
    values = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    max_weight = int(input())

    max_value, chosen = knapsack(values, weights, max_weight)

    print("The max value possible is")
    print(max_value)

    print("The index chosen for these are")
    print(' '.join(str(x) for x in chosen))

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
# -------------------------- FCSN CLASS ---------------------------
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
# ----------------------- GENERATE SUMMARY ----------------------
####################################################################

parser = argparse.ArgumentParser(description='Generate keyshots, keyframes and score bar.')
parser.add_argument('--h5_path', type=str, help='path to hdf5 file that contains information of a dataset.',
                    default='../data/fcsn_tvsum.h5')
parser.add_argument('-j', '--json_path', type=str,
                    help='path to json file that stores pred score output by model, it should be saved in score_dir.',
                    default='score_dir/epoch-49.json')
parser.add_argument('-r', '--data_root', type=str, help='path to directory of original dataset.',
                    default='../data/TVSum')
parser.add_argument('-s', '--save_dir', type=str, help='path to directory where generating results should be saved.',
                    default='Results')
parser.add_argument('-b', '--bar', action='store_true', help='whether to plot score bar.')

args = parser.parse_args()
h5_path = args.h5_path
json_path = args.json_path
data_root = args.data_root
save_dir = args.save_dir
bar = args.bar
video_dir = os.path.join(data_root, 'ydata-tvsum50-v1_1', 'video')
anno_path = os.path.join(data_root, 'ydata-tvsum50-v1_1', 'data', 'ydata-tvsum50-anno.tsv')
f_data = h5py.File(h5_path)
with open(json_path) as f:
    json_dict = json.load(f)
    ids = json_dict.keys()


def get_keys(id):
    video_info = f_data['video_' + id]
    video_path = os.path.join(video_dir, id + '.mp4')
    cps = video_info['change_points'][()]
    pred_score = json_dict[id]['pred_score']
    pred_selected = json_dict[id]['pred_selected']

    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    frames = np.array(frames)
    keyshots = []
    for sel in pred_selected:
        for i in range(cps[sel][0], cps[sel][1]):
            keyshots.append(frames[i])
    keyshots = np.array(keyshots)

    write_path = os.path.join(save_dir, id, 'summary.avi')
    video_writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*'XVID'), 24, keyshots.shape[2:0:-1])
    for frame in keyshots:
        video_writer.write(frame)
    video_writer.release()

    keyframe_idx = [np.argmax(pred_score[cps[sel][0]: cps[sel][1]]) + cps[sel][0] for sel in pred_selected]
    keyframes = frames[keyframe_idx]

    keyframe_dir = os.path.join(save_dir, id, 'keyframes')
    os.mkdir(keyframe_dir)
    for i, img in enumerate(keyframes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(os.path.join(keyframe_dir, '{}.jpg'.format(i)))


def plot_bar():
    with open(anno_path) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        csv_dict = {}
        idx = 0
        for row in csv_reader:
            score = np.array([int(i) for i in row[2].split(',')])
            if str(idx // 20 + 1) in ids:
                if idx % 20 == 0:
                    csv_dict[str(idx // 20 + 1)] = score / 20
                else:
                    csv_dict[str(idx // 20 + 1)] += score / 20
            idx += 1

    sns.set()
    fig, ax = plt.subplots(ncols=1, nrows=len(ids), figsize=(30, 20))
    fig.tight_layout()
    for id, axi in zip(ids, ax.flat):
        scores = csv_dict[id]
        pred_summary = json_dict[id]['pred_summary']
        axi.bar(left=list(range(len(scores))), height=scores, color=['lightseagreen' if i == 0
                                                                     else 'orange' for i in pred_summary],
                edgecolor=None)
        axi.set_title(id)
    save_path = os.path.join(save_dir, 'result-bar.png')
    plt.savefig(save_path)


def gen_summary():
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for id in ids:
        os.mkdir(os.path.join(save_dir, id))
        get_keys(id)

    if bar:
        plot_bar()


if __name__ == '__main__':
    plt.switch_backend('agg')
    gen_summary()

f_data.close()

####################################################################
# ----------------------- TRAIN CLASS ----------------------
####################################################################

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable



class Solver(object):
    """Class that Builds, Trains FCSN model"""

    def __init__(self, config=None, train_loader=None, test_dataset=None):
        self.config = config
        self.train_loader = train_loader
        self.test_dataset = test_dataset

        # model
        self.model = FCSN(self.config.n_class)

        # optimizer
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters())
            # self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=self.config.momentum)
            self.model.train()

        if self.config.gpu:
            self.model = self.model.cuda()

        if not os.path.exists(self.config.score_dir):
            os.mkdir(self.config.score_dir)

        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

    @staticmethod
    def sum_loss(pred_score, gt_labels, weight=None):
        n_batch, n_class, n_frame = pred_score.shape
        log_p = torch.log_softmax(pred_score, dim=1).reshape(-1, n_class)
        gt_labels = gt_labels.reshape(-1)
        criterion = torch.nn.NLLLoss(weight)
        loss = criterion(log_p, gt_labels)
        return loss

    def train(self):
        writer = SummaryWriter(log_dir=self.config.log_dir)
        t = trange(self.config.n_epochs, desc='Epoch', ncols=80)
        for epoch_i in t:
            sum_loss_history = []

            for batch_i, (feature, label, _) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):

                # [batch_size, 1024, seq_len]
                feature.requires_grad_()
                # => cuda
                if self.config.gpu:
                    feature = feature.cuda()
                    label = label.cuda()

                # ---- Train ---- #
                pred_score = self.model(feature)

                label_1 = label.sum() / label.shape[0]
                label_0 = label.shape[1] - label_1
                weight = torch.tensor([label_1, label_0], dtype=torch.float)

                if self.config.gpu:
                    weight = weight.cuda()

                loss = self.sum_loss(pred_score, label, weight)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                sum_loss_history.append(loss)

            mean_loss = torch.stack(sum_loss_history).mean().item()
            t.set_postfix(loss=mean_loss)
            writer.add_scalar('Loss', mean_loss, epoch_i)

            if (epoch_i + 1) % 5 == 0:
                ckpt_path = self.config.save_dir + '/epoch-{}.pkl'.format(epoch_i)
                tqdm.write('Save parameters at {}'.format(ckpt_path))
                torch.save(self.model.state_dict(), ckpt_path)
                self.evaluate(epoch_i)
                self.model.train()

    def evaluate(self, epoch_i):
        self.model.eval()
        out_dict = {}
        eval_arr = []
        table = PrettyTable()
        table.title = 'Eval result of epoch {}'.format(epoch_i)
        table.field_names = ['ID', 'Precision', 'Recall', 'F-score']
        table.float_format = '1.3'

        with h5py.File(self.config.data_path) as data_file:
            for feature, label, idx in tqdm(self.test_dataset, desc='Evaluate', ncols=80, leave=False):
                if self.config.gpu:
                    feature = feature.cuda()
                pred_score = self.model(feature.unsqueeze(0)).squeeze(0)
                pred_score = torch.softmax(pred_score, dim=0)[1]
                video_info = data_file['video_' + str(idx)]
                pred_score, pred_selected, pred_summary = eval.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()]
                eval_res = [eval.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr]
                eval_res = np.mean(eval_res, axis=0).tolist()

                eval_arr.append(eval_res)
                table.add_row([idx] + eval_res)

                out_dict[idx] = {
                    'pred_score': pred_score,
                    'pred_selected': pred_selected, 'pred_summary': pred_summary
                }

        score_save_path = self.config.score_dir + '/epoch-{}.json'.format(epoch_i)
        with open(score_save_path, 'w') as f:
            tqdm.write('Save score at {}'.format(str(score_save_path)))
            json.dump(out_dict, f)
        eval_mean = np.mean(eval_arr, axis=0).tolist()
        table.add_row(['mean'] + eval_mean)
        tqdm.write(str(table))


if __name__ == '__main__':
    train_config = Config()
    test_config = Config(mode='test')
    train_loader, test_dataset = get_loader(train_config.data_path, batch_size=train_config.batch_size)
    solver = Solver(train_config, train_loader, test_dataset)
    solver.train()

####################################################################
# ------------------------- MAKE DATASETS -------------------------
####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, help='directory containing mp4 file of specified dataset.',
                    default='../data/TVSum_video')
parser.add_argument('--h5_path', type=str, help='save path of the generated dataset, which should be a hdf5 file.',
                    default='../data/fcsn_tvsum.h5')
parser.add_argument('--vsumm_data', type=str,
                    help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.',
                    default='../data/eccv_datasets/eccv16_dataset_tvsum_google_pool5.h5')

args = parser.parse_args()
video_dir = args.video_dir
h5_path = args.h5_path
vsumm_data = h5py.File(args.vsumm_data)


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

net = models.googlenet(pretrained=True).float().cuda()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])


def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8);
        recall = overlap / (true_sum + 1e-8);
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    tqdm.write('Overlap: ' + str(overlap_arr))
    tqdm.write('True summary n_key: ' + str(true_sum_arr))
    tqdm.write('Oracle smmary n_key: ' + str(oracle_sum))
    tqdm.write('Final F-score: ' + str(best_fscore))
    return oracle_summary


def video2fea(video_path, h5_f):
    video = cv2.VideoCapture(video_path.as_uri())
    idx = video_path.as_uri().split('.')[0].split('/')[-1]
    tqdm.write('Processing video ' + idx)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length // 320
    fea = []
    label = []
    usr_sum_arr = vsumm_data['video_' + idx]['user_summary'][()]
    usr_sum = get_oracle_summary(usr_sum_arr)
    cps = vsumm_data['video_' + idx]['change_points'][()]
    n_frame_per_seg = vsumm_data['video_' + idx]['n_frame_per_seg'][()]
    i = 0
    success, frame = video.read()
    while success:
        if (i + 1) % ratio == 0:
            fea.append(fea_net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu())
            try:
                label.append(usr_sum[i])
            except:
                pdb.set_trace()
        i += 1
        success, frame = video.read()
    fea = torch.stack(fea)
    fea = fea[:320]
    label = label[:320]
    v_data = h5_f.create_group('video_' + idx)
    v_data['feature'] = fea.numpy()
    v_data['label'] = label
    v_data['length'] = len(usr_sum)
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    v_data['picks'] = [ratio * i for i in range(320)]
    v_data['user_summary'] = usr_sum_arr
    if fea.shape[0] != 320 or len(label) != 320:
        print('error in video ', idx, feashape[0], len(label))


def make_dataset(video_dir, h5_path):
    video_dir = Path(video_dir).resolve()
    video_list = list(video_dir.glob('*.mp4'))
    video_list.sort()
    with h5py.File(h5_path, 'w') as h5_f:
        for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
            video2fea(video_path, h5_f)


if __name__ == '__main__':
    make_dataset(video_dir, h5_path)

vsumm_data.close()