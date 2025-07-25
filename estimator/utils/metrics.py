import numpy as np
from scipy.stats import norm


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs(np.divide((true - pred), true, where=true!=0)))


def MSPE(pred, true):
    return np.mean(np.square(np.divide((true - pred), true, where=true!=0)))


def quan_recall(mu, sigma, true, p):
    pred_num = norm.ppf(p, loc=mu, scale=sigma)
    return np.sum(pred_num > true) / (true.shape[0]*true.shape[1]*true.shape[2])


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "MSPE": mspe
    }


def metric_with_recall(pred_mu, pred_sigma, true):
    mae = MAE(pred_mu, true)
    mse = MSE(pred_mu, true)
    rmse = RMSE(pred_mu, true)
    mape = MAPE(pred_mu, true)
    mspe = MSPE(pred_mu, true)
    recall_90 = quan_recall(pred_mu, pred_sigma, true,0.9)
    recall_95 = quan_recall(pred_mu, pred_sigma, true,0.95)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "MSPE": mspe,
        "RECALL_0.9": recall_90,
        "RECALL_0.95": recall_95
    }
