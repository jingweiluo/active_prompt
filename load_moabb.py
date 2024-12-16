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

    n_components = 8  # 选择要分解的主成分数量
    csp = CSP(n_components=n_components, norm_trace=False)

    def get_freq_spatial_feat(X, labels, csp):
        feat_map = []
        for lowcut in range(8,29,4):
            X_filtered = bandpass_filter(X, lowcut=lowcut, highcut=lowcut+4, fs=250)
            X_csp = csp.fit_transform(X_filtered, labels)
            X_csp_expanded = X_csp[:, np.newaxis, :]
            feat_map.append(X_csp_expanded)
        return np.concatenate(feat_map, axis=1)

    feat_map = get_freq_spatial_feat(X, labels, csp) # num_trials, n_components * num_freq_bands 由于n_components < n_chans, 所以=n_chans
    index = get_first_index(meta, column_index=1, target_value="3test") # number of training trials

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

    train_data, test_data, train_labels, test_labels = np.delete(feat_map, testid_dict[testid], axis=0), feat_map[testid_dict[testid]], np.delete(labels, testid_dict[testid], axis=0), labels[testid_dict[testid]]
    print('数据的shape为:', train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    return train_data, test_data, train_labels, test_labels
    # np.savez('moabb_bci_iv_2b', train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels)

# get_moabb_data("2b", 1, 3)
