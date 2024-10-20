# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import pickle
#
# # 载入嵌入数据
# with open(f'initial_emb_wd', 'rb') as f:
#     emb_init = pickle.load(f)
#
# with open(f'final_emb_wd', 'rb') as f:
#     emb_final = pickle.load(f)
#
# # 将数据展平为 2D，合并维度 0 和 1
# emb_init_flat = np.vstack([emb_init[:, 0, :], emb_init[:, 1, :]])
# emb_final_flat = np.vstack([emb_final[:, 0, :], emb_final[:, 1, :]])
#
# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# emb_init_2d = tsne.fit_transform(emb_init_flat)  # 初始嵌入降维
# emb_final_2d = tsne.fit_transform(emb_final_flat)  # 最终嵌入降维
#
# # 如果你想用 PCA 代替 t-SNE，可以这样做：
# # pca = PCA(n_components=2)
# # emb_init_2d = pca.fit_transform(emb_init_flat)
# # emb_final_2d = pca.fit_transform(emb_final_flat)
#
# # 绘制初始嵌入的散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(emb_init_2d[:, 0], emb_init_2d[:, 1], color='blue', alpha=0.5, label='Initial Embedding')
#
# # 绘制最终嵌入的散点图
# plt.scatter(emb_final_2d[:, 0], emb_final_2d[:, 1], color='red', alpha=0.5, label='Final Embedding')
#
# plt.title('t-SNE Visualization of Embeddings')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.legend()
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import pickle

with open(f'initial_emb_wd', 'rb') as f:
    emb_init = pickle.load(f)

with open(f'final_emb_wd', 'rb') as f:
    emb_final = pickle.load(f)


emb_init_0 = emb_init[:, 0, :]
emb_init_1 = emb_init[:, 1, :]
cos_sim_init = cdist(emb_init_0, emb_init_1, metric='cosine')
cos_sim_init = 1 - np.diag(cos_sim_init)

emb_final_0 = emb_final[:, 0, :]
emb_final_1 = emb_final[:, 1, :]
cos_sim_final = cdist(emb_final_0, emb_final_1, metric='cosine')
cos_sim_final = 1 - np.diag(cos_sim_final)


# KDE 密度估计
kde_init = gaussian_kde(cos_sim_init)
kde_final = gaussian_kde(cos_sim_final)

# 生成 X 轴数据 (余弦相似度范围)
x_vals = np.linspace(-1, 1, 500)

# 计算初始和最终的 PDF
pdf_init = kde_init(x_vals)
pdf_final = kde_final(x_vals)

# 绘制概率密度函数
plt.figure(figsize=(8, 6))
plt.plot(x_vals, pdf_init, label='Initial Embedding', color='blue', linestyle='--')
plt.plot(x_vals, pdf_final, label='Final Embedding', color='red')
plt.fill_between(x_vals, pdf_init, alpha=0.3, color='blue')
plt.fill_between(x_vals, pdf_final, alpha=0.3, color='red')
plt.title('Cosine Similarity Probability Density Function')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
