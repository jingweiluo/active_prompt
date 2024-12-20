
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

def findIndex(X, v):
    found_idx = None
    for i in range(X.shape[0]):
        if np.array_equal(X[i], v):  # 检查每个切片是否与 v 相同
            found_idx = i
            break
    if found_idx is None:
        print("未在X中找到匹配的向量v")
    return found_idx

# ==========================================================载入数据==================================================
sub_index = 5
train_data, test_data, train_labels, test_labels = get_moabb_data("2a", sub_index, 0) # 2b数据集，sub_index, test_id
lab_map = {
    'left_hand': 1,
    'right_hand':2
}
X = train_data.reshape(train_data.shape[0],-1)
targets = np.array([lab_map[lab] for lab in train_labels])

Y = test_data.reshape(test_data.shape[0],-1)
targets_y = np.array([lab_map[lab] for lab in test_labels])


def qbc(n_members, n_initial, n_queries):
    # ==========================================================初始化数据==================================================
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 12
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

    # =============================================================Query=====================================================
    idx_list = []
    # n_queries = 8
    for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        gloabl_idx = findIndex(X, query_instance[0])
        idx_list.append(gloabl_idx)
        committee.teach(
            X=X_pool[query_idx],
            y=y_pool[query_idx]
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

    # =============================================================测试=====================================================
    test_score = committee.score(Y, targets_y)
    # print('测试集准确率:', test_score)
    with open(f"log/qbc_log", 'a',  encoding='utf-8') as file:
        file.write(f"测试集准确率: {test_score}\n")
    return test_score


def rand_select(n_members, n_initial, n_queries):
    # ==========================================================初始化数据==================================================
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 12
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
    # =============================================================测试=====================================================
    test_score = committee.score(Y, targets_y)
    # print('测试集准确率:', test_score)
    with open(f"log/random_log", 'a',  encoding='utf-8') as file:
        file.write(f"测试集准确率: {test_score}\n")
    return test_score


qbc_score_list = []
random_score_list = []

for i in range(30):
    qbc_score = qbc(n_members=5, n_initial=12, n_queries=8)
    qbc_score_list.append(qbc_score)
    random_score = qbc(n_members=5, n_initial=20, n_queries=0)
    random_score_list.append(random_score)

qbc_mean = np.mean(qbc_score_list)
random_mean = np.mean(random_score_list)
print(qbc_mean, random_mean)




# =============================================================画图=====================================================

# tsne降维
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# label_to_color={
#     1: 'grey',
#     2: 'yellow'
# }

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







