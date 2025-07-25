import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_covariates = configs.num_covariates
        self.num_class = configs.num_class
        self.class_output_dim = configs.class_output_dim
        self.linear_input = self.seq_len + (self.num_covariates + self.num_class) * self.class_output_dim
        self.pred_len = configs.pred_len

        # Embedding and Attention
        self.embedding = nn.ModuleList()
        for i in range(self.num_class):
            self.embedding.append(nn.Embedding(configs.class_input_dim[i], self.class_output_dim))

        self.time_embedding = nn.ModuleList()
        self.time_embedding.append(nn.Embedding(2, self.class_output_dim))  # 工作日
        self.time_embedding.append(nn.Embedding(7, self.class_output_dim))  # 周几
        self.time_embedding.append(nn.Embedding(24, self.class_output_dim))  # 小时

        self.q = nn.Linear(self.class_output_dim, configs.attn_hidden_dim)
        self.k = nn.Linear(self.class_output_dim, configs.attn_hidden_dim)
        self.v = nn.Linear(self.class_output_dim, self.class_output_dim)
        self._norm_fact = 1 / math.sqrt(configs.attn_hidden_dim)

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.linear_input, self.pred_len)
            self.Linear_Trend = nn.Linear(self.linear_input, self.pred_len)
            self.Linear_Sigma = nn.Linear(self.linear_input, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.linear_input]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.linear_input]))
            self.Linear_Sigma.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.linear_input]))

        self.distribution_sigma = nn.Softplus()

    def encoder(self, x, x_dec, idx):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        attention_input = torch.zeros((x.shape[0], x.shape[2], self.num_class, self.class_output_dim),
                                      device=x.device)
        for i in range(self.num_class):
            attention_input[:, :, i, :] = self.embedding[i](idx[:, i, :]) # batch_size * organization_num * num_class * class_output_dim

        Q = self.q(attention_input)  # Q: batch_size * organization_num * num_class * attn_hidden_dim
        K = self.k(attention_input)  # K: batch_size * organization_num * num_class * attn_hidden_dim
        V = self.v(attention_input)  # V: batch_size * organization_num * num_class * class_output_dim

        attention_output = torch.zeros((x.shape[0], x.shape[2], self.num_class*self.class_output_dim),
                                       device=x.device)
        for i in range(x.shape[0]):
            atten = nn.Softmax(dim=-1)(
                torch.bmm(Q[i], K[i].permute(0, 2, 1))) * self._norm_fact  # batch_size * num_class * num_class

            attention_output[i] = torch.bmm(atten, V[i]).view(x.shape[2], -1)

        time_features = torch.zeros((x.shape[0], x.shape[2], self.num_covariates, self.class_output_dim),
                                   device=x.device)
        for i in range(self.num_covariates):
            time_features[:, :, i, :] = self.time_embedding[i](x_dec[:, :, i].to(torch.long))
        time_features = time_features.view(x.shape[0], x.shape[2], -1)

        seasonal_input = torch.cat((seasonal_init, time_features, attention_output), dim=2)
        trend_input = torch.cat((trend_init, time_features, attention_output), dim=2)
        sigma_input = torch.cat((x.permute(0, 2, 1), time_features, attention_output), dim=2)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_input)
            trend_output = self.Linear_Trend(trend_input)
            presigma = self.Linear_Sigma(sigma_input)
        x = seasonal_output + trend_output
        sigma_output = self.distribution_sigma(presigma)
        return x.permute(0, 2, 1), sigma_output.permute(0, 2, 1)

    def forward(self, x_enc, x_dec, idx, mask=None):
        mu_out, sigma_out = self.encoder(x_enc, x_dec, idx)
        return mu_out, sigma_out, torch.zeros(1, device=x_enc.device)  # [B, L, D]
