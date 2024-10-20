import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
import pickle
# 假设 emb_init 和 emb_final 分别为初始和最终的 embedding
# 它们的形状为 (1397, 2, 1536)
with open(f'initial_emb_wd', 'rb') as f:
    emb_init = pickle.load(f)

with open(f'final_emb_wd', 'rb') as f:
    emb_final = pickle.load(f)
# 计算每个锚点在维度 0 和 1 上的余弦相似度
cos_sim_init = np.array([1 - cosine(emb_init[i, 0, :], emb_init[i, 1, :]) for i in range(emb_init.shape[0])])
cos_sim_final = np.array([1 - cosine(emb_final[i, 0, :], emb_final[i, 1, :]) for i in range(emb_final.shape[0])])

# 确保余弦相似度值在 [-1, 1] 的合理范围内
cos_sim_init = np.clip(cos_sim_init, -1, 1)
cos_sim_final = np.clip(cos_sim_final, -1, 1)

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
