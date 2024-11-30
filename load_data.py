import scipy.io # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

datapath_SEED_IV_eeg_feature = 'e:/EData/SEED_IV/eeg_feature_smooth/'
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