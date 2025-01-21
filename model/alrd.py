import warnings
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, \
    recall_score, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor

warnings.simplefilter("ignore")
class ALRD():
    def __init__(self, X_pool, y_pool, budget):
        self.X = X_pool
        self.y = y_pool
        # 按照2:8的比例划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=42)

        self.X_pool = X_train # 训练数据
        self.y_pool = y_train # 训练集标签
        self.X_test = X_test # 测试数据
        self.y_test = y_test # 测试集标签
        self.nSample, self.nAtt = self.X_pool.shape # 样本数量和特征数量
        self.labeled = self.init_labeled() # 已经标记的数据索引
        self.labels = np.array([i for i in range(1)])  # 初始聚类的簇编号集合，大小等于其特征数量
        self.unlabeled = self.init_unlabeled() # 所有未标记的数据索引
        self.nClass = len(self.labels) # 聚类的簇大小
        self.budgetLeft = deepcopy(budget) # 待选择的样本
        self.model = AdaBoostRegressor() # 模型

    def init_labeled(self):
        labeled = []
        nCluster = self.X_pool.shape[1]
        y_pred = KMeans(n_clusters=nCluster).fit_predict(self.X_pool) # 聚类操作，为类的大小为特征大小
        cluster_labels, count = np.unique(y_pred, return_counts=True)
        cluster_dict = OrderedDict()  # 定义有序字典

        # 初始化每个簇对应一个集合
        for i in cluster_labels:  # 遍历簇，形成一个集合
            cluster_dict[i] = []  # 每个簇用预测结果 i 代表
        for i in range(len(y_pred)):
            cluster_dict[y_pred[i]].append(i)  # 将每个集合都放入

        for i in cluster_labels:
            centroid = np.mean(self.X_pool[cluster_dict[i]], axis=0) # 计算各簇的中心位置
            close_dist = np.inf
            tar_idx = None
            for idx in cluster_dict[i]:# 找到距离最近样本点
                if np.linalg.norm(self.X_pool[idx] - centroid) < close_dist:
                    close_dist = np.linalg.norm(self.X_pool[idx] - centroid)
                    tar_idx = idx
            labeled.append(tar_idx)
        # print(labeled)

        return labeled
    def init_unlabeled(self):
        unlabeled = [i for i in range(self.nSample)] # 遍历所有样本
        for idx in self.labeled:
            unlabeled.remove(idx) # 去掉已经标记的样本
        return unlabeled

    def evaluation(self):
        """
        使用 K 折交叉验证，K 中的验证集作为测试集，取平均作为泛化误差
        :return:
        """
        X_train = self.X_pool[self.labeled]
        y_train = self.y_pool[self.labeled]
        self.model.fit(X_train, y_train)

        y_pred_test = self.model.predict(self.X_test)  # 测试集预测
        rmse = np.sqrt(np.mean((y_pred_test - self.y_test) ** 2))
        # print("测试集rmse：", round(rmse, 8))

        # # 在所有数据集上进行 K 折，将验证集作为测试集
        # AL_Train_scores = []
        # AL_Test_scores = []
        # AL_R2_scores = []
        # kfold = KFold(n_splits=10, shuffle=True).split(self.X_pool, self.y_pool)
        # for k, (train, test) in enumerate(kfold):
        #     AL_Train_score = mean_squared_error(self.model.predict(self.X_pool[train]), self.y_pool[train])
        #     AL_Test_score = mean_squared_error(self.model.predict(self.X_pool[test]),self.y_pool[test])
        #     AL_R2_score = r2_score(self.model.predict(self.X_pool[test]), self.y_pool[test])
        #
        #     AL_Train_scores.append(AL_Train_score)
        #     AL_Test_scores.append(AL_Test_score)
        #     AL_R2_scores.append(AL_R2_score)

        # AL_Train_MSE = np.mean(AL_Train_scores)
        # AL_Test_MSE = np.mean(AL_Test_scores)
        # AL_R2 = np.mean(AL_R2_scores)
        # print('训练集 MAE：', AL_Train_MSE, '测试集 MAE：', AL_Test_MSE)

        # return AL_Train_MSE, AL_Test_MSE, AL_R2
        return rmse

    def select(self):
        while self.budgetLeft > 0:
            nCluster = len(self.labeled) + 1
            y_pred = KMeans(n_clusters=nCluster).fit_predict(self.X_pool) # 聚类操作，类的大小为初始样本数 + 1
            cluster_labels, count = np.unique(y_pred, return_counts=True)
            cluster_dict = OrderedDict() # 定义有序字典

            # 初始化每个簇对应一个集合
            for i in cluster_labels: # 遍历簇，形成一个集合
                cluster_dict[i] = [] # 每个簇用预测结果 i 代表

            # 先将已经标记的样本加入到对应的簇中
            for idx in self.labeled:
                cluster_dict[y_pred[idx]].append(idx) # 将已经标记好的样本添加进入
            empty_ids = OrderedDict() # 定义一个空字典

            for i in cluster_labels: # 遍历簇中心
                if len(cluster_dict[i]) == 0: # 判断此时簇是否为空，如果含有标记样本则为空，没有标记样本则不为空
                    empty_ids[i] = count[i] # 记录空簇的未标记样本的数量

            # 在所有没有任何提前标记的样本的簇中，即 empty_ids 中，选择簇包含样本数量最大的对应的簇编号
            tar_label = max(empty_ids, key=empty_ids.get) #
            tar_cluster_ids = [] # 存储 tar_label 簇中的样本索引
            # 将在 tar_label 簇中的样本加入到集合 tar_cluster_ids 中
            for idx in range(self.nSample): # 遍历样本数量
                if y_pred[idx] == tar_label: # 判断样本是否在 tar_label 中
                    tar_cluster_ids.append(idx)
            centroid = np.mean(self.X_pool[tar_cluster_ids], axis=0) # 计算 tar_cluster_ids 的中心位置

            tar_idx = None
            close_dist = np.inf

            # 计算 tar_cluster_ids 中，所有样本到簇中心的距离，选择距离最近的一个样本
            for idx in tar_cluster_ids:
                if np.linalg.norm(self.X_pool[idx] - centroid) < close_dist:
                    close_dist = np.linalg.norm(self.X_pool[idx] - centroid)
                    tar_idx = idx

            # 将选出的样本进行标记
            self.labeled.append(tar_idx)
            self.unlabeled.remove(tar_idx)
            self.budgetLeft -= 1

        return self.labeled

def plot_scatter(labeled, xpool, ypool):
    pca = PCA(n_components=2).fit_transform(xpool)
    colors = ['blue'] * xpool.shape[0]  # 其他点为蓝色
    # 将选中的样本点的颜色设置为红色
    for idx in labeled:
        colors[idx] = 'red'
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=colors)
    # 设置图形标题
    plt.title("Visualization of Selected and Unselected Points")
    # 显示图形
    plt.show()

if __name__ == '__main__':

    # budget = 40
    X, y = make_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
    # plot_scatter([], X, y)

    budget_set = 30
    rmse = np.zeros((budget_set, 20))
    for budget in range(1, budget_set+1):
        for i in range(20):
            model = ALRD(X_pool=X, y_pool=y, budget=budget)
            labeled = model.select()
            print(labeled)
            rmse[budget-1, i] = model.evaluation()
            print(f"当budget为{budget}时，且进行到第{i}次，测试集rmse为{rmse[budget-1, i] }")

    rmse_mean = np.mean(rmse, axis=1)
    plt.plot(range(len(rmse_mean)), rmse_mean)
    plt.show()
    # plot_scatter(labeled, X, y)

    # 训练集全部训练，看一看上限
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_base = AdaBoostRegressor()
    model_base.fit(X_train, y_train)
    y_pred_test = model_base.predict(X_test)  # 测试集预测
    rmse_base = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    print(f"上限rmse为{rmse_base}")

