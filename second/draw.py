import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your data in DataFrame as before
anchor_similarities = pd.read_csv(r'B:/6254_HW/second/wd_anchor_similarities.csv')['Similarity']
node_similarities = pd.read_csv(r'B:/6254_HW/second/wd_node_similarities.csv')['Similarity']

plt.figure(figsize=(10, 6))

# Density plot for anchor similarities
sns.kdeplot(anchor_similarities, shade=True, label='Anchor Similarities')

# Density plot for node similarities
sns.kdeplot(node_similarities, shade=True, label='Node Similarities', color='red')

plt.title('Density Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Density')
plt.legend()

plt.show()
