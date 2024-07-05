import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO

# Function to create a subplot for the heatmap
def create_subplot(data, title, subplot_position, fig):
    df = pd.read_csv(StringIO(data), index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    ax = fig.add_subplot(subplot_position)
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlOrRd", cbar=True, linewidths=.5, ax=ax)
    ax.title.set_text(title)

# Create a figure with 4 subplots
fig = plt.figure(figsize=(20, 16))
titles = ['DBLP_1', 'DBLP_2', 'WD']
positions = [221, 222, 223]

# Actual data for each dataset
data1 = """
, None, shuffle, cutoff, word-repetition
single, 0.9657, 0.9713, 0.9613, 0.9720
add_shuffle, , ,0.9705, 0.9673
add_cutoff, , , ,0.9703
all, , , , 0.9759
"""
data3 = """
, None, shuffle, cutoff, word-repetition
single, 0.6499, 0.6537, 0.6528, 0.6486
add_shuffle, , ,0.6570, 0.6543
add_cutoff, , , ,0.6540
all, , , , 0.6593
"""
data4 = """
, None, shuffle, cutoff, word-repetition
single, 0.4685, 0.4741, 0.4807, 0.4739
add_shuffle, , ,0.4720, 0.4810
add_cutoff, , , ,0.4781
all, , , , 0.4774
"""

# List of data for iteration
data_sets = [data1, data3, data4]

# Iterate over each dataset and create a subplot for each one
for title, pos, data in zip(titles, positions, data_sets):
    create_subplot(data, title, pos, fig)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure with all subplots
plt.savefig("heatmaps.pdf")
# Show the figure with all subplots
plt.show()
