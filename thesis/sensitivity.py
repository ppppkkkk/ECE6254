import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Improved code with functions and better organization

def plot_data(ax, x, y_data, x_label, y_label, title):
    """
    Function to plot data on a given axes.
    """
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(labelsize=fontsize)
    y_ticks = np.arange(0.4, 1.0, 0.2)
    x_ticks = np.arange(1, 5, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_label)
    ax.set_ylabel(y_label, size=fontsize)
    ax.set_xlabel('Feature Dimension', size=fontsize)

    for i, (method, values) in enumerate(y_data.items()):
        ax.plot(x, values, marker=marker_pool[i], c=color_pool[i], markersize=10)

    ax.grid(ls='--')

# Common parameters and data
fontsize = 24
color_pool = ['#C82423', '#9AC9DB', '#8E44AD', '#EC7063', '#2878B5', '#999999',  '#14517C']
marker_pool = ['o', '^', 'v', 'h', '*', '<', '>', 's']
x = [1, 2, 3, 4]
x_label = ['60', '80', '100', '120']

# Data for the plots
y_data1 = {'$WD$': [0.4291, 0.4218, 0.4152, 0.4176], 'DBLP_1': [0.6040, 0.6082, 0.6106, 0.6174],
           'DBLP_2': [0.9598, 0.9574, 0.9564, 0.9616]}

y_data2 = {'$WD$': [0.4859, 0.4798, 0.4744, 0.4765], 'DBLP_1': [0.6403, 0.6483, 0.6509, 0.6586],
           'DBLP_2': [0.9732, 0.9714, 0.9711, 0.9743]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot data
plot_data(ax1, x, y_data1, x_label, '$top$@1', '(a) $top$@1 vs $\lambda$')
plot_data(ax2, x, y_data2, x_label, '$MRR$', '(b) $MRR$ vs $\lambda$')

# Adjust layout
fig.tight_layout(pad=2, w_pad=0.4)

# Adjusting legend position above the plot area
lgd = fig.legend(bbox_to_anchor=(0.5, 1.12), labels=y_data1.keys(), loc='upper center', borderaxespad=0.10,
                 prop={'size': 20}, ncol=6)

# Save the plot
plt.savefig('parameters_DM_adjusted_legend_above.pdf')



