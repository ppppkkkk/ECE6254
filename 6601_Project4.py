import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

directory = r'C:/Users/11792/Desktop/6669/project4'
significance_level = 0.05

combined_data = pd.DataFrame()

for file in os.listdir(directory):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(directory, file))
        df['First_Difference'] = df['Close'].diff() / df['Close'].shift(1) * 100
        df['Stock'] = file.replace('.csv', '')  # Use the file name as the stock identifier
        combined_data = combined_data._append(df[['Date', 'Stock', 'First_Difference']], ignore_index=True)

adf_results = {}
non_random_count = 0
for stock in combined_data['Stock'].unique():
    series = combined_data[combined_data['Stock'] == stock]['First_Difference'].dropna()
    adf_test = adfuller(series)
    adf_results[stock] = adf_test[1]
    if adf_test[1] < significance_level:
        non_random_count += 1

total_stocks = len(combined_data['Stock'].unique())

random_percentage = ((total_stocks - non_random_count) / total_stocks) * 100
non_random_percentage = (non_random_count / total_stocks) * 100

summary_stats = {
    'Random Percentage': random_percentage,
    'Non-Random Percentage': non_random_percentage,
    'Non-Random Count': non_random_count,
    'Total Stocks': total_stocks
}

summary_df = pd.DataFrame([summary_stats])
adf_results_df = pd.DataFrame(list(adf_results.items()), columns=['Stock', 'p-value'])

print("Summary Statistics:")
print(summary_df)

print("\np-values for Each Stock:")
print(adf_results_df)
