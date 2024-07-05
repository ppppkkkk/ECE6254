import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 初始数据
initial_investment = -5e6
initial_revenue = 1.5e6
initial_cost = 0.5e6
years = np.arange(6)

# 定义各个情景
scenarios = {
    'Optimistic': {'revenue_growth': 0.10, 'cost_growth': 0.05, 'discount_rate': 0.05},
    'Neutral': {'revenue_growth': 0.08, 'cost_growth': 0.06, 'discount_rate': 0.06},
    'Pessimistic': {'revenue_growth': 0.06, 'cost_growth': 0.07, 'discount_rate': 0.07}
}

# 创建DataFrame
sensitivity_df = pd.DataFrame(index=list(scenarios.keys()), columns=['NPV ($)', 'Most Likely Scenario'])

# 计算各个情景下的NPV
for scenario, parameters in scenarios.items():
    revenues = [initial_revenue * (1 + parameters['revenue_growth']) ** year for year in years]
    costs = [initial_cost * (1 + parameters['cost_growth']) ** year for year in years]
    net_cash_flows = np.array(revenues) - np.array(costs)
    discounted_cash_flows = net_cash_flows / (1 + parameters['discount_rate']) ** years
    npv = discounted_cash_flows.sum() + initial_investment
    sensitivity_df.loc[scenario, 'NPV ($)'] = npv

# 假设"Neutral"情景最有可能
sensitivity_df['Most Likely Scenario'] = ['Yes' if index == 'Neutral' else 'No' for index in sensitivity_df.index]

# 使用matplotlib创建表格图片
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.axis('tight')
ax.table(cellText=sensitivity_df.values, colLabels=sensitivity_df.columns, rowLabels=sensitivity_df.index, cellLoc='center', loc='center')
ax.set_title('Sensitivity Analysis: NPV under Different Scenarios', fontsize=15, y=0.7) # y=1.08 调整标题位置

plt.tight_layout()
plt.savefig('Sensitivity_Analysis_with_Likelihood.png')
plt.show()
