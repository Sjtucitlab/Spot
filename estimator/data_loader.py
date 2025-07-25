import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib
from torch.utils.data import DataLoader, Dataset, Sampler
from chinese_calendar import is_workday

matplotlib.use('Agg')


class Dataset_GPU_Org(Dataset):
    def __init__(self, args, data=None, timestamp=None,  flag='train',
                 size=None, scale=False, timeenc=0, freq='h'):
        self.args = args
        self.val_days = 7
        self.stride_size = 5
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.cov_dicts = []
        self.vc_key_dicts = {}
        self.inverse_vc_key_dicts = {}
        self.scaler = StandardScaler()
        self.timestamp = timestamp
        self.__read_data__(data)

    def __read_data__(self, df_raw):
        df_raw.index = pd.DatetimeIndex(df_raw.index)

        train_start_timestamp = pd.Timestamp("2024-03-01 20:00:00")
        val_start_time = self.timestamp - pd.Timedelta(days=self.val_days, hours=self.seq_len)
        val_end_time = self.timestamp - pd.Timedelta(minutes=1)
        train_end_time = val_end_time - pd.Timedelta(days=self.val_days)

        border1s = [train_start_timestamp, val_start_time]
        border2s = [train_end_time, val_end_time]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.copy()

        self.data_stamp = df_raw[border1:border2].index
        self.class_data = self.fit_class(df_raw.columns)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data[border1:border2].values)
        else:
            data = df_data[border1:border2].values

        self.data_x, self.data_y = data, data


    def __getitem__(self, index):
        s_begin = index*self.stride_size
        s_end = s_begin + self.seq_len*60
        seq_x = self.data_x[s_begin:s_end]
        seq_x = np.max(seq_x.reshape(int(seq_x.shape[0] / 60), 60, seq_x.shape[1]), axis=1)

        seq_x_mark = self.get_covariates(self.data_stamp[s_end])
        seq_x_mark = np.expand_dims(seq_x_mark, 1).repeat(seq_x.shape[1], axis=1).T

        r_begin = s_end - self.label_len * 60
        r_end = r_begin + (self.label_len + self.pred_len) * 60
        seq_y = self.data_y[r_begin:r_end]
        seq_y = np.max(seq_y.reshape(int(seq_y.shape[0] / 60), 60, seq_y.shape[1]), axis=1)

        return seq_x, seq_x_mark, self.class_data.T, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len*60 - self.pred_len*60) // self.stride_size + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def fit_covariates(self, times):
        num_covariates = self.args.num_covariates
        covariates = np.zeros((times.shape[0], num_covariates))
        for i, input_time in enumerate(times):
            covariates[i, 0] = is_workday(input_time)
            covariates[i, 1] = input_time.weekday()
            covariates[i, 2] = input_time.hour
        for i in range(num_covariates):
            self.cov_dicts.append(dict(zip(covariates[:, i], stats.zscore(covariates[:, i]))))

    def get_covariates(self, time):
        covariates = np.zeros(self.args.num_covariates)
        covariates[0] = is_workday(time)
        covariates[1] = time.weekday()
        covariates[2] = time.hour
        return covariates

    def fit_class(self, columns):
        for i in range(self.args.num_class):
            keys = pd.Series([vc_name[i] for vc_name in columns.str.split('&&', expand=True)])
            self.vc_key_dicts[i] = dict(zip(keys.unique(), range(keys.nunique())))
            self.inverse_vc_key_dicts[i] = dict(zip(range(keys.nunique()), keys.unique()))
        values = []
        for col in columns:
            col_values = []
            key = col.split('&&')
            for i in range(self.args.num_class):
                col_values.append(self.vc_key_dicts[i][key[i]])
            values.append(col_values)
        return np.array(values)
