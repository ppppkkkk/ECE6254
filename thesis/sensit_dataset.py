import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# Extracting top@1 values for each dataset and method for the proportions
top1_data = {
    'DBLP_1': {
        'ADIE_UIL': [0.6571, 0.6429, 0.6114, 0.5475, 0.3844],
        'MAUIL': [0.6368, 0.6076, 0.5714, 0.4941, 0.3158],
        'Grad-Align': [0.6847, 0.6441, 0.5953, 0.5413, 0.4898],
        'GAlign': [0.8114, 0.8135, 0.7756, 0.8014, 0.8140],
        'CENALP': [0.0832, 0.0856, 0.0893, 0.0890, 0.0764],
        'Deeplink': [0.0621, 0.0465, 0.0403, 0.0221, 0.0098]
    },

    'DBLP_2': {
        'ADIE_UIL': [0.9720, 0.9700, 0.9561, 0.9369, 0.8767],
        'MAUIL': [0.6220, 0.7646, 0.8265, 0.8649, 0.8897],
        'Grad-Align': [0.3801, 0.3509, 0.3344, 0.2838, 0.2210],
        'GAlign': [0.3009, 0.2989, 0.2585, 0.2540, 0.2711],
        'CENALP': [0.0836, 0.0791, 0.0735, 0.0712, 0.0707],
        'Deeplink':[0.0433, 0.0347, 0.0267, 0.0206, 0.0135]
    },

    'WD': {
        'ADIE_UIL': [0.4900, 0.4514, 0.4093, 0.3207, 0.1516],
        'MAUIL': [0.3937, 0.3452, 0.3018, 0.2211, 0.0970],
        'Grad-Align': [0.5340, 0.4865, 0.4044, 0.3099, 0.3271],
        'GAlign': [0.0215, 0.0274, 0.0215, 0.0233, 0.0167],
        'CENALP': [0.0017, 0.0021, 0.0009, 0.0011, 0.0008],
        'Deeplink':[0.0057, 0.0024, 0.0020, 0.0018, 0.0015]
    }

}

mrr_data = {
    'DBLP_1': {
        'ADIE_UIL': [0.6985, 0.6831, 0.6524, 0.5977, 0.4512],
        'MAUIL': [0.6897, 0.6594, 0.6260, 0.5531, 0.3871],
        'Grad-Align': [0.7497, 0.7177, 0.6782, 0.6300, 0.5892],
        'GAlign': [0.8116, 0.8136, 0.7758, 0.8016, 0.8142],
        'CENALP': [0.3771, 0.3878, 0.3998, 0.4059, 0.3577],
        'Deeplink': [0.0630, 0.0472, 0.0412, 0.0231, 0.0107]
    },

    'DBLP_2': {
        'ADIE_UIL': [0.9821, 0.9801, 0.9712, 0.9568, 0.9133],
        'MAUIL': [0.9164, 0.8943, 0.8632, 0.8129, 0.6941],
        'Grad-Align': [0.4661, 0.4400, 0.4209, 0.3748, 0.3036],
        'GAlign': [0.3006, 0.3044, 0.2574, 0.2568, 0.2718],
        'CENALP': [0.3673, 0.3528, 0.3436, 0.3423, 0.3292],
        'Deeplink':[0.0439, 0.0353, 0.0272, 0.0211, 0.0140]
    },

    'WD': {
        'ADIE_UIL': [0.5416, 0.5080, 0.4721, 0.3902, 0.2223],
        'MAUIL': [0.4567, 0.4124, 0.3617, 0.2892, 0.1521],
        'Grad-Align': [0.5522, 0.5075, 0.4233, 0.3325, 0.3632],
        'GAlign': [0.0223, 0.0283, 0.0224, 0.0241, 0.0175],
        'CENALP': [0.0268, 0.0285, 0.0201, 0.0174, 0.0153],
        'Deeplink':[0.0066, 0.0033, 0.0030, 0.0027, 0.0023]
    }

}

# Proportions for x-axis
proportions = ['0.5', '0.6', '0.7', '0.8', '0.9']

# Set a more contrasting color palette
colors = ['#C82423', '#9AC9DB', '#8E44AD', '#EC7063', '#2878B5',  '#14517C']

markers = ['o', 'v', '^', '<', '>', 's']
# Create a 4x2 subplot layout
# 调整图例位置使其位于整个图形的最上方

# 设置图形大小和子图布局
fig, axs = plt.subplots(3, 2, figsize=(10, 14), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

# 绘制左侧子图的 top@1 数据
for i, (dataset, methods) in enumerate(top1_data.items()):
    ax = axs[i, 0]
    for j, (method, top1_values) in enumerate(methods.items()):
        ax.plot(proportions, top1_values, marker=markers[j], label=method, color=colors[j])
    ax.set_title(f'top@1 on {dataset}')
    ax.set_xlabel('Proportion of anchor links')
    ax.set_ylabel('top@1')
    ax.grid(True)

# 绘制右侧子图的 MRR 数据
for i, (dataset, methods) in enumerate(mrr_data.items()):
    ax = axs[i, 1]
    for j, (method, mrr_values) in enumerate(methods.items()):
        ax.plot(proportions, mrr_values, marker=markers[j], label=method, color=colors[j])
    ax.set_title(f'MRR on {dataset}')
    ax.set_xlabel('Proportion of anchor links')
    ax.set_ylabel('MRR')
    ax.grid(True)

# 创建带有不同标记的统一图例
lines = [mlines.Line2D([], [], color=colors[i], marker=markers[i], label=list(top1_data['DBLP_1'].keys())[i]) for i in range(len(top1_data['DBLP_1']))]
fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=len(top1_data['DBLP_1']), fontsize='large')

# 调整布局和间距
fig.subplots_adjust(top=0.9)
# 显示图形


# Save the plots to a file
top1_plot_path = 'top1_comparison_contrastive_plots.pdf'
plt.savefig(top1_plot_path)

