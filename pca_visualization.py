import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.manifold import TSNE # type: ignore
from load_data import load_all_trials

# 加载数据
train_data, test_data, y, test_labels = load_all_trials(7)
X = np.array(train_data)
X = X.reshape(X.shape[0], -1)

# PCA降维到2维
# pca = PCA(n_components=2)
# X_compressed = pca.fit_transform(X)

# 使用 t-SNE 将数据降维到2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_compressed     = tsne.fit_transform(X)

# 绘制所有样本点
plt.figure(figsize=(18, 6))

# 高亮随机选择的样本
random_indices = [0, 1, 2, 3, 4, 5, 6, 7]
plt.subplot(1,3,1)
plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c='gray', label='Samples', alpha=0.7)
plt.scatter(
    X_compressed[random_indices, 0], 
    X_compressed[random_indices, 1], 
    c='red', 
    label='Random Selected Samples', 
    s=100, 
    edgecolors='black',
    alpha=0.5
)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization with Selected Samples')
plt.legend()

# 高亮主动选择的样本
active_indices = [1, 7, 13, 23, 31, 42, 51, 53]
plt.subplot(1,3,2)
plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c='gray', label='Samples', alpha=0.7)
plt.scatter(
    X_compressed[active_indices, 0], 
    X_compressed[active_indices, 1], 
    c='blue', 
    label='Active Selected Samples', 
    s=100, 
    edgecolors='black',
    alpha=0.5
)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization with Selected Samples')
plt.legend()

plt.subplot(1,3,3)
plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c=y, cmap='viridis', s=50)
plt.colorbar(label='Classes')  # 添加颜色条
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization with Selected Samples')

# 添加图例和标题
plt.show()