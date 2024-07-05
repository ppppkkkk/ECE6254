import numpy as np
import matplotlib.pyplot as plt

# 颜色和标记的配置
color_pool = ['#2878B5', '#9AC9DB', '#F8AC8C', '#14517C', '#999999', '#8E44AD', '#EC7063', '#C82423']
marker_pool = ['o', '^', 'h', 'v', '*', '<', '>', 's']

title = "ab"

fontsize = 24
font2 = {'family': 'Times New Roman', 'size': 20}
font3 = {'family': 'Times New Roman', 'size': 25}

# 创建只有两个子图的布局
fig, (ax, ax2) = plt.subplots(1, 2)
fig.set_size_inches(12, 5)

x_ticks = ['WD', 'DBLP_1', 'DBLP_2']
labels = ['EFC-UIL-CS', 'EFC-UIL-CT', 'EFC-UIL-TS', 'EFC-UIL']

# 第一个子图
ax.set_ylim(0.0, 1.0)
ax.tick_params(labelsize=fontsize)

y0 = [0.3974, 0.1852, 0.469]
y1 = [0.4902, 0.5936, 0.482]
y2 = [0.042, 0.1522, 0.1292]
y3 = [0.4196, 0.6192, 0.9542]

xth = np.arange(len(x_ticks))
ax.tick_params(labelsize=16)
width = 0.7

# 绘制柱状图
ax.bar(4 * xth, y0, width, facecolor='#C82423', edgecolor='#000000')
ax.bar(4 * xth + width, y1, width, facecolor='#9AC9DB', edgecolor='#000000')
ax.bar(4 * xth + 2 * width, y2, width, facecolor='#2878B5', edgecolor='#000000')
ax.bar(4 * xth + 3 * width, y3, width, facecolor='#EC7063', edgecolor='#000000')

ax.set_ylabel('top@1', fontsize=20)
ax.set_xticks(4 * xth + 1)
ax.set_xticklabels(x_ticks)

# 第二个子图
ax2.set_ylim(0.0, 1.0)
ax2.tick_params(labelsize=16)

y10 = [0.4597, 0.2764, 0.5548]
y11 = [0.5307, 0.6027, 0.5127]
y12 = [0.0878, 0.2409, 0.2105]
y13 = [0.4774, 0.6593, 0.9759]

# 绘制柱状图
ax2.bar(4 * xth, y10, width, facecolor='#C82423', edgecolor='#000000')
ax2.bar(4 * xth + width, y11, width, facecolor='#9AC9DB', edgecolor='#000000')
ax2.bar(4 * xth + 2 * width, y12, width, facecolor='#2878B5', edgecolor='#000000')
ax2.bar(4 * xth + 3 * width, y13, width, facecolor='#EC7063', edgecolor='#000000')

ax2.set_ylabel('MRR', fontsize=20)
ax2.set_xticks(4 * xth + 1)
ax2.set_xticklabels(x_ticks)

# 图例和布局调整
ax.legend(labels, loc='upper right', ncol=4, bbox_to_anchor=(2.1, 1.2), prop={'size': 20})
plt.subplots_adjust(right=1.1, top=0.9)

# 保存图片
plt.savefig(title + '.png', format='png', dpi=200, bbox_inches='tight')
