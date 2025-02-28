"""
================================
Tutorial 1: Simple Motor Imagery
================================

In this example, we will go through all the steps to make a simple BCI
classification task, downloading a dataset and using a standard classifier. We
choose the dataset 2a from BCI Competition IV, a motor imagery task. We will
use a CSP to enhance the signal-to-noise ratio of the EEG epochs and a LDA to
classify these signals.
"""

# Authors: Pedro L. C. Rodrigues, Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery

import numpy as np


moabb.set_log_level("info")
warnings.filterwarnings("ignore")

from scipy.signal import butter, filtfilt
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    对信号进行频带滤波。
    
    参数:
    - data: 输入的信号 (1D 或 2D 数组)。
    - lowcut: 低频截止频率 (Hz)。
    - highcut: 高频截止频率 (Hz)。
    - fs: 采样率 (Hz)。
    - order: 滤波器阶数 (默认为 4)。

    返回:
    - 滤波后的信号 (与输入 shape 相同)。
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 设计滤波器
    b, a = butter(order, [low, high], btype='band')
    
    # 应用滤波器
    filtered_data = filtfilt(b, a, data, axis=-1)  # 支持多维数据，沿最后一维滤波
    return filtered_data

def get_first_index(dataframe, column_index, target_value):
    """
    获取 DataFrame 的第 column_index 列中，第一个等于 target_value 的索引。

    参数:
    - dataframe: pandas.DataFrame，数据表
    - column_index: int，要查询的列索引（从 0 开始）
    - target_value: str，要查找的值

    返回:
    - 第一个匹配项的索引，如果不存在返回 None
    """
    # 获取第 column_index 列的数据
    column_data = dataframe.iloc[:, column_index]

    # 查找第一个匹配项的索引
    result = column_data[column_data == target_value].index
    return result[0] if not result.empty else None


##############################################################################
# Instantiating Dataset
# ---------------------
#
# The first thing to do is to instantiate the dataset that we want to analyze.
# MOABB has a list of many different datasets, each one containing all the
# necessary information for describing them, such as the number of subjects,
# size of trials, names of classes, etc.
#
# The dataset class has methods for:
#
# - downloading its files from some online source (e.g. Zenodo)
# - importing the data from the files in whatever extension they might be
#   (like .mat, .gdf, etc.) and instantiate a Raw object from the MNE package
def get_moabb_data(dataset_name, sub_index, testid):
    """
    获取MI训练及测试数据
    参数:
    - dataframe(string): 数据集名称 ~["2a", "2b"]
    - sub_index(int): 被试编号 ~[1-9]
    - testid(int): 测试集索引 ~[0, 1, 2, 3, 4]
    返回:
    - train_data(list[np(6*3)]): 特征值
    - train_labels(list[string]): 标签 
    """
    dataset = BNCI2014_001() if dataset_name == "2a" else BNCI2014_004()
    paradigm = LeftRightImagery()
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[sub_index])

    #============================这一部分是提取CSP特征==========================================
    n_components = 2  # 选择要分解的主成分数量
    csp = CSP(n_components=n_components, norm_trace=False)

    def get_freq_spatial_feat(X, labels, csp):
        feat_map = []
        for lowcut in range(8,29,4):
            X_filtered = bandpass_filter(X, lowcut=lowcut, highcut=lowcut+4, fs=250)
            X_csp = csp.fit_transform(X_filtered, labels)
            X_csp_expanded = X_csp[:, np.newaxis, :]
            feat_map.append(X_csp_expanded)
        return np.concatenate(feat_map, axis=1)

    feat_map = get_freq_spatial_feat(X, labels, csp) # num_trials, num_freq_bands, n_components 由于n_components < n_chans, 所以=n_chans
    feat_map = feat_map[:, 0:4, :]
    index = get_first_index(meta, column_index=1, target_value="1test") # number of training trials
    print(meta)

    # #============================这一部分是将raw-EEG投影到image-EEG的共空间==========================================
    # import os
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    # from torch import Tensor
    # from einops.layers.torch import Rearrange
    # X = X[:,:,250:500] # (288, 22, 1000)

    # chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
	# 			  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
	# 			  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
	# 			  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
	# 			  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
	# 			  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
	# 			  'O1', 'Oz', 'O2']
    # channel_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',  'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',  'T9', 'T10']

    # expand_X = np.zeros((X.shape[0], 63, X.shape[2]))
    # # 对应chan_order的顺序插入，无值的补0
    # # for i, c in enumerate(channel_list):
    # #     try:
    # #         index = chan_order.index(c)  # 假设 'orange' 不在列表中
    # #         expand_X[:, index, :] = X[:, i, :]
    # #     except ValueError:
    # #         print(f"{c} is not in the list")

    # # 直接补0
    # for i, c in enumerate(channel_list):
    #     expand_X[:, i, :] = X[:, i, :]

    # model_idx = 'test0'
    # gpus = [0,4,5,6,7]
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

    # class ResidualAdd(nn.Module):
    #     def __init__(self, fn):
    #         super().__init__()
    #         self.fn = fn

    #     def forward(self, x, **kwargs):
    #         res = x
    #         x = self.fn(x, **kwargs)
    #         x += res
    #         return x

    # class PatchEmbedding(nn.Module):
    #     def __init__(self, emb_size=40):
    #         super().__init__()
    #         # revised from shallownet
    #         self.tsconv = nn.Sequential(
    #             nn.Conv2d(1, 40, (1, 25), (1, 1)),
    #             nn.AvgPool2d((1, 51), (1, 5)),
    #             nn.BatchNorm2d(40),
    #             nn.ELU(),
    #             nn.Conv2d(40, 40, (63, 1), (1, 1)),
    #             nn.BatchNorm2d(40),
    #             nn.ELU(),
    #             nn.Dropout(0.5),
    #         )

    #         self.projection = nn.Sequential(
    #             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
    #             Rearrange('b e (h) (w) -> b (h w) e'),
    #         )

    #     def forward(self, x: Tensor) -> Tensor:
    #         # b, _, _, _ = x.shape
    #         x = self.tsconv(x)
    #         x = self.projection(x)
    #         return x

    # class FlattenHead(nn.Sequential):
    #     def __init__(self):
    #         super().__init__()

    #     def forward(self, x):
    #         x = x.contiguous().view(x.size(0), -1)
    #         return x


    # class Enc_eeg(nn.Sequential):
    #     def __init__(self, emb_size=40, **kwargs):
    #         super().__init__(
    #             PatchEmbedding(emb_size),
    #             FlattenHead()
    #         )

            
    # class Proj_eeg(nn.Sequential):
    #     def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
    #         super().__init__(
    #             nn.Linear(embedding_dim, proj_dim),
    #             ResidualAdd(nn.Sequential(
    #                 nn.GELU(),
    #                 nn.Linear(proj_dim, proj_dim),
    #                 nn.Dropout(drop_proj),
    #             )),
    #             nn.LayerNorm(proj_dim),
    #         )

    # Enc_eeg = Enc_eeg().cuda()
    # Proj_eeg = Proj_eeg().cuda()
    # Enc_eeg = nn.DataParallel(Enc_eeg, device_ids=[i for i in range(len(gpus))])
    # Proj_eeg = nn.DataParallel(Proj_eeg, device_ids=[i for i in range(len(gpus))])

    # model_base_url = '/home/luojingwei/code/NICE-EEG'
    # Enc_eeg.load_state_dict(torch.load(model_base_url + '/model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
    # Proj_eeg.load_state_dict(torch.load(model_base_url + '/model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)

    # Enc_eeg.eval()
    # Proj_eeg.eval()
    
    # expand_X = torch.from_numpy(expand_X)
    # expand_X = expand_X.float()
    # expand_X = expand_X.unsqueeze(1)

    # print(expand_X.shape)

    # tfea = Proj_eeg(Enc_eeg(expand_X))
    # tfea = tfea / tfea.norm(dim=1, keepdim=True)

    # feat_map = tfea
    # feat_map = F.avg_pool1d(tfea.unsqueeze(1), kernel_size=10, stride=10).squeeze(1)
    # feat_map = feat_map.cpu().detach().numpy()

    # ==================== 公共部分：切分数据集 ========================================
    if dataset_name == "2a":
        # 2a数据集分成了2个session，在不同天采集，分别由144，144个trial
        s1 = slice(0, 144)
        s2 = slice(144, 288)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "2b":
        # 2b数据集分成了5个session，分别由120，120，160，160， 160个trial
        s1 = slice(0, 120)
        s2 = slice(120, 240)
        s3 = slice(240, 400)
        s4 = slice(400, 560)
        s5 = slice(560, 720)

        testid_dict = {
            0: s1,
            1: s2,
            2: s3,
            3: s4,
            4: s5
        }
    else: raise ValueError("该数据集不存在")

    train_data, test_data, train_labels, test_labels = np.delete(feat_map, testid_dict[testid], axis=0), feat_map[testid_dict[testid]], np.delete(labels, testid_dict[testid], axis=0), labels[testid_dict[testid]]
    print('数据的shape为:', train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    return train_data, test_data, train_labels, test_labels

# get_moabb_data("2a", 3, 1)
