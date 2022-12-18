import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, xs, labels, edge_index, mode='train', config=None, normalizer=None):
        # self.raw_data = raw_data
        self.xs = xs
        self.labels = labels

        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.normalizer = normalizer

        if mode != 'train':
            assert normalizer is not None

        data = [torch.tensor(x).double() for x in xs]
        labels = [torch.tensor(label).double() for label in labels]

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)

    def process(self, xs, labels):  # modify here
        # instances: List of time-series instances -> have to be a list of torch.tensor(data).double()
        # data: 2d-array = [xs1, xs2, ...] where xs1 = feature_1 for all data
        # row = features, col = timestamps
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        # node_num, total_time_len = data.shape # n_features, n_timesteps

        # rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        # for data in data_lst
        for data, label in zip(xs, labels):
            node_num, total_time_len = data.shape
            rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
            for i in rang:

                ft = data[:, i-slide_win:i]  # select all features for i-slide_win:i timesteps
                # row = features, column = time
                tar = data[:, i]  # multivariate timeseries to be predicted

                x_arr.append(ft)
                y_arr.append(tar)
                # x_arr = [data_all_feat1, data_all_feat2, ...]   data_all_feat1: row = features, column = time

                labels_arr.append(label[i])  # abnormal/normal labels

        # each x = [x1, x2, ...] where x1 = sliding window of size slide_win

        #should normalize here
        # x_arr = np.array(x_arr)  # row = features, column = time
        xs_flatten = []
        if self.mode == 'train':
            for x_time_window in x_arr:  # for time
                # for i in range(len(x_time_window[0])):
                #     print(x_time_window[:, i])
                xs_flatten += [list(x_time_window[:, i]) for i in range(len(x_time_window[0]))]  # loop over time index
            normalizer = MinMaxScaler(feature_range=(0, 1)).fit(xs_flatten)
            self.normalizer = normalizer

        x_arr = [torch.Tensor(self.normalizer.transform(xx.T).T.tolist()).double() for xx in x_arr]
        y_arr = [torch.Tensor(self.normalizer.transform([yy.tolist()])[0]).double() for yy in y_arr]

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





