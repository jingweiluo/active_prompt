import sys
import os
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_parent_dir)
import numpy as np
# RANDOM_STATE_SEED = 1
# np.random.seed(RANDOM_STATE_SEED)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from active_prompt.load.load_moabb import get_moabb_data
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling, consensus_entropy_sampling
from scipy.stats import entropy
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans

def findIndex(X, v):
    found_idx = None
    for i in range(X.shape[0]):
        if np.array_equal(X[i], v):  # 检查每个切片是否与 v 相同
            found_idx = i
            break
    if found_idx is None:
        print("未在X中找到匹配的向量v")
    return found_idx

def qbc(train_data, train_labels, n_members, n_initial, n_queries, init_idx=None):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    targets = np.array([lab_map[lab] for lab in train_labels])
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 12
    n_idx = init_idx if init_idx is not None else np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)

    for member_idx in range(n_members):
        train_idx = np.random.choice(n_idx, size=n_initial, replace=True)

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, n_idx, axis=0)
    y_pool = np.delete(y_pool, n_idx)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=consensus_entropy_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )

    # =============================================================Query=====================================================
    idx_list = []
    # n_queries = 8
    for _ in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        gloabl_idx = findIndex(X, query_instance[0])
        idx_list.append(gloabl_idx)
        committee.teach(
            X=X_pool[query_idx],
            y=y_pool[query_idx]
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
    return idx_list, committee

def rand_select(train_data, train_labels, n_members, n_initial):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    targets = np.array([lab_map[lab] for lab in train_labels])
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 20
    n_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)

    for member_idx in range(n_members):
        train_idx = np.random.choice(n_idx, size=n_initial, replace=True)

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, n_idx, axis=0)
    y_pool = np.delete(y_pool, n_idx)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=consensus_entropy_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )
    return n_idx, committee

# =============================================================测试=====================================================
def compare_test(train_data, test_data, train_labels, test_labels, n_times, init_idx):
    # 导入测试数据
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    Y = test_data.reshape(test_data.shape[0],-1)
    targets_y = np.array([lab_map[lab] for lab in test_labels])

    rd_qbc_score_list = []
    qbc_score_list = []
    rand_score_list = []

    for i in tqdm(range(n_times)):
        _, rd_qbc_committe = qbc(train_data, train_labels, n_members=5, n_initial=12, n_queries=8, init_idx=init_idx)
        rd_qbc_score = rd_qbc_committe.score(Y, targets_y)
        rd_qbc_score_list.append(rd_qbc_score)
    
        _, qbc_committe = qbc(train_data, train_labels, n_members=5, n_initial=12, n_queries=8)
        qbc_score = qbc_committe.score(Y, targets_y)
        qbc_score_list.append(qbc_score)

        _, rand_committe = rand_select(train_data, train_labels, n_members=5, n_initial=20)
        rand_score = rand_committe.score(Y, targets_y)
        rand_score_list.append(rand_score)

        rd_qbc_mean = np.mean(rd_qbc_score_list)
        qbc_mean = np.mean(qbc_score_list)
        random_mean = np.mean(rand_score_list)
        print(rd_qbc_mean, qbc_mean, random_mean)

def find_init_centroids(X, init_num):
    '''
    返回k个cluster中，离质心最近的样本点的index
    '''
    kmeans = KMeans(init_num)
    kmeans.fit(X)
    # 获取初始质心
    centroids = kmeans.cluster_centers_
    # 为每个数据点分配聚类标签
    labels = kmeans.labels_

    # 找到每个聚类中心最近的样本点
    nearest_points = []
    for i in range(centroids.shape[0]):
        # 计算当前聚类中所有点到聚类中心的距离
        distances = np.linalg.norm(X[labels == i] - centroids[i], axis=1)
        # 找到最小距离的索引
        min_index = np.argmin(distances)
        # 使用该索引获取最近的点
        nearest_point = X[labels == i][min_index]
        nearest_points.append(nearest_point)

    init_id_list = []
    for sample in nearest_points:
        gloabl_idx = findIndex(X, sample)
        init_id_list.append(gloabl_idx)
    print(init_id_list)
    return init_id_list

    
if __name__=='__main__':
    # ==========================================================载入数据==================================================
    dataset_name = "2a"
    sub_index = 5
    test_id = 1
    repeat_times = 1

    train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, sub_index, test_id)

    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    train_data = np.array(train_data)
    X = train_data.reshape(train_data.shape[0],-1)
    targets = np.array([lab_map[lab] for lab in train_labels])

    test_data = np.array(test_data)
    Y = test_data.reshape(test_data.shape[0],-1)
    targets_y = np.array([lab_map[lab] for lab in test_labels])

    init_id_list = find_init_centroids(X, init_num=12)

    compare_test(train_data, test_data, train_labels, test_labels, repeat_times, init_idx=init_id_list)


# =============================================================画图=====================================================

    # # tsne降维
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(X)

    # label_to_color={
    #     1: 'grey',
    #     2: 'yellow'
    # }

    # plt.figure(figsize=(2 * 7, 7))
    # plt.subplot(1, 2, 1)
    # plt.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=[label_to_color[tar] for tar in targets], cmap='viridis', s=50)
    # plt.scatter(x=X_tsne[init_id_list, 0], y=X_tsne[init_id_list, 1], c='red', s=20, label='Highlighted')
    # plt.show()

    # plt.figure(figsize=((n_members+1) * 7, 7))
    # plt.subplot(1, n_members+1, 1)
    # plt.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=[label_to_color[tar] for tar in targets], cmap='viridis', s=50)
    # plt.scatter(x=X_tsne[idx_list, 0], y=X_tsne[idx_list, 1], c='red', s=20, label='Highlighted')
    # for idx in idx_list:
    #     plt.text(X_tsne[idx, 0], X_tsne[idx, 1], f'Index {idx}', fontsize=10, ha='right', color='red')
    # plt.title('ground truth label and qbc queried samples')

    # for learner_idx, learner in enumerate(committee):
    #     plt.subplot(1, n_members+1, learner_idx + 2)
    #     plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=[label_to_color[tar] for tar in learner.predict(X)], cmap='viridis', s=50)
    #     plt.scatter(x=X_tsne[idx_list, 0], y=X_tsne[idx_list, 1], c='red', s=20, label='Highlighted')
    #     for idx in idx_list:
    #         plt.text(X_tsne[idx, 0], X_tsne[idx, 1], f'Index {idx}', fontsize=10, ha='right', color='red')
    #     plt.title('Learner no. %d initial predictions' % (learner_idx + 1))

    # plt.show()
