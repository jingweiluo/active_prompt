import scipy.io # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import mne # type: ignore
import matplotlib.pyplot as plt # type: ignore

datapath_SEED_IV_eeg_feature = 'e:/Data/SEED_IV/eeg_feature_smooth/'
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
all_labels = session1_label + session2_label + session3_label

def load_all_trials(subject_index):
    all_trials = []
    for session in range(1,4):
        # load subject data
        eeg_datafile = datapath_SEED_IV_eeg_feature + str(subject_index) + '_' + str(session) + '.mat'
        eeg_data = scipy.io.loadmat(eeg_datafile)

        for i in range(1, 24+1):
            eeg_feature_name = 'de_LDS' + str(i)
            eeg_trial = eeg_data[eeg_feature_name]
            selected_chans_trial = eeg_trial[[14, 22, 23, 31, 32, 40], :, :].mean(axis=1) # 选择指定的6个通道
            all_trials.append(selected_chans_trial)

    train_data, test_data, train_labels, test_labels = train_test_split(all_trials, all_labels, test_size=0.2, random_state=42)
    return train_data, test_data, train_labels, test_labels

def load_all_segs(subject_index):
    all_trials = []
    all_segs = []
    all_segs_labels = []

    for session in range(1,4):
        # load subject data
        eeg_datafile = datapath_SEED_IV_eeg_feature + str(subject_index) + '_' + str(session) + '.mat'
        eeg_data = scipy.io.loadmat(eeg_datafile)

        for i in range(1, 24+1):
            eeg_feature_name = 'de_LDS' + str(i)
            eeg_trial = eeg_data[eeg_feature_name]
            selected_chans_trial = eeg_trial[[14, 22, 23, 31, 32, 40], :, :] # 选择指定的6个通道
            all_trials.append(selected_chans_trial)
    
    for (i, tri) in enumerate(all_trials):
        for j in range(tri.shape[1]):
            all_segs.append(tri[:, j, :])
            all_segs_labels.append(all_labels[i])
    
    train_data, test_data, train_labels, test_labels = train_test_split(all_segs, all_segs_labels, test_size=0.2, random_state=42)
    return train_data, test_data, train_labels, test_labels

# MI Dataset===========================================================================================================
"""
Extract task-related EEG trials from raw signal.

Args:
    arg1 (int): Subject index, from 1 to 9

Returns:
    train_data(list): list of 3 * 1000 ndArray

可修改点：
(1) 划分数据集方法是按照原论文中方法, 前三个为训练集, 后两个为测试集。也可从随机点开始选择一定比例(60%)的连续样本作为训练集
(2) 去掉了EOG眼动信号
(3) 没做滤波,可以1-40或者4-40
(4) mne.Epochs的入参proj为true,也可为False
(5) tmin, tmax = 3, 7
"""
def load_bci_iv_2b(sub_index):
    datapath_bci_iv_2b = 'e:/Data/BCICIV_2b_gdf/'
    trials = []
    labels = []
    sep_index = 0

    for i in range(1,6):
        filename = f"B0{sub_index}0{i}T" if i < 4 else f"B0{sub_index}0{i}E"
        trial_path = datapath_bci_iv_2b + filename + '.gdf'
        label_path = datapath_bci_iv_2b + 'true_labels/' + filename + '.mat'

        if i == 4:
            sep_index = len(trials)
        
        trials.extend(get_bci_2b_trials(trial_path))
        labels.extend(get_bci_2b_labels(label_path))

    # Use the first three sessions as training data, and the last two sessions as testing data.
    train_data, test_data, train_labels, test_labels = trials[:sep_index], trials[sep_index:], labels[:sep_index], labels[sep_index:]
    # print(len(train_data), len(test_data), len(train_labels), len(test_labels))
    return train_data, test_data, train_labels, test_labels

def get_bci_2b_trials(filepath):
    raw = mne.io.read_raw_gdf(filepath, preload=True)
    # raw.filter(1, 40)

    # extract events and event_id
    events, event_id = mne.events_from_annotations(raw)

    # eliminate EOG channels
    raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']

    # return Indices of good channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    # start and end seconds of each trial
    tmin, tmax = 3, 7

    # event_id to retain
    keys_to_keep = ['769', '770', '783']
    filtered_ids = {key: event_id[key] for key in keys_to_keep if key in event_id}

    epochs = mne.Epochs(raw, events, filtered_ids, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_data = epochs.get_data()
    epochs_data = epochs_data[:, :, :-1]
    epochs_list = [epochs_data[i] for i in range(epochs_data.shape[0])]
    return epochs_list

def get_bci_2b_labels(filepath):
    mat_data = scipy.io.loadmat(filepath)
    labels = list(mat_data['classlabel'].reshape(-1))
    return labels

# load_bci_iv_2b(9)