import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from estimator.utils.losses import likelihood_loss
from estimator.utils.metrics import metric_with_recall
from estimator.utils.tools import adjust_learning_rate, EarlyStopping
from estimator.model import OrgLinear
from estimator.data_loader import Dataset_GPU_Org
from chinese_calendar import is_workday


def data_provider(args, flag, data, timestamp):
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Dataset_GPU_Org(
        args=args,
        flag=flag,
        data=data,
        timestamp=timestamp,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        scale=args.scaling
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


class GPURequestEstimator:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.cov_dicts = []
        self.class_data = np.array([])

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = OrgLinear.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, data, timestamp):
        data_set, data_loader = data_provider(self.args, flag, data, timestamp)
        if flag == "train":
            self.class_data = data_set.class_data.T
            self.cov_dicts = data_set.cov_dicts
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = likelihood_loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, idx, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                idx = idx.to(torch.long).to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                output_mu, output_sigma, _ = self.model(batch_x, batch_x_mark, idx)

                output_mu = output_mu[:, -self.args.pred_len:, :]
                output_sigma = output_sigma[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                output_mu = output_mu.detach().cpu()
                output_sigma = output_sigma.detach().cpu()
                batch_y = batch_y.detach().cpu()

                if vali_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    output_mu = vali_data.inverse_transform(output_mu.reshape(shape[0] * shape[1], -1)).reshape(
                        shape)
                    batch_y = vali_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                loss = criterion(output_mu, output_sigma, batch_y)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, data, now_timestamp):
        train_data, train_loader = self._get_data(flag='train', data=data, timestamp=now_timestamp)
        vali_data, vali_loader = self._get_data(flag='val', data=data, timestamp=now_timestamp)

        path = os.path.join(self.args.checkpoints, str(now_timestamp))
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_mark, idx, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                idx = idx.to(torch.long).to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                output_mu, output_sigma, loss = self.model(batch_x, batch_x_mark, idx)

                output_mu = output_mu[:, -self.args.pred_len:, :]
                output_sigma = output_sigma[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss += criterion(output_mu, output_sigma, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, data, now_timestamp):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_x_mark, idx = self.get_test_data(data, now_timestamp)
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            idx = idx.to(torch.long).to(self.device)

            # encoder - decoder
            output_mu, output_sigma, _ = self.model(batch_x, batch_x_mark, idx)

            output_mu = output_mu[:, -self.args.pred_len:, :]
            output_sigma = output_sigma[:, -self.args.pred_len:, :]
            output_mu = output_mu.detach().cpu().numpy()
            output_sigma = output_sigma.detach().cpu().numpy()
            # if test_data.scale and self.args.inverse:
            #     shape = batch_y.shape
            #     output_mu = test_data.inverse_transform(output_mu.reshape(shape[0] * shape[1], -1)).reshape(shape)
            #     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            pred_mu = output_mu[:, :, :]
            pred_sigma = output_sigma[:, :, :]

        # print(pred_mu, pred_sigma)

        return pred_mu, pred_sigma

    def get_test_data(self, data, timestamp):
        seq_x = data.values[-self.args.seq_len*60:]
        seq_x = np.max(seq_x.reshape(int(seq_x.shape[0] / 60), 60, seq_x.shape[1]), axis=1)

        seq_x_mark = self.get_covariates(timestamp)
        seq_x_mark = np.expand_dims(seq_x_mark, 1).repeat(seq_x.shape[1], axis=1).T

        return (torch.from_numpy(seq_x).unsqueeze(0), torch.from_numpy(seq_x_mark).unsqueeze(0),
                torch.from_numpy(self.class_data).unsqueeze(0))

    def get_covariates(self, time):
        covariates = np.zeros(self.args.num_covariates)
        covariates[0] = is_workday(time)
        covariates[1] = time.weekday()
        covariates[2] = time.hour
        return covariates
