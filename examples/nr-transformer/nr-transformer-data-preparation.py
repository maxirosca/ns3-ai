import pandas as pd
import numpy as np
import torch
from scipy.special import softmax


# Load the CSV file
df = pd.read_csv('/home/maximilianrosca/Masterarbeit/Simulation-files/dl_weights_log.csv')

# Create a new column for total_schedule_round
x = []
x.append(0)
for i in range(1, len(df.index)):
    if i % 2 == 0:
        x.append(x[i-1] + 1)
    else:
        x.append(x[i-1])
df = df.assign(total_schedule_round=pd.Series(x))

# Remove rows where 'avg_tput' is less than or equal to 0
df = df[(df['avg_tput'] > 0)]

# Remove rows where 'schedule_round' is 1
df = df[df['schedule_round'] != 1]

# Remove rows where weight is suspiciously high (outliers)
# df = df[df['weight'] < 1e+6]

# Sample a fixed number of schedule rounds
n = 50000  # Number of schedule rounds to sample
unique_rounds = df['total_schedule_round'].unique()
sampled_rounds = np.random.choice(unique_rounds, size=n, replace=False)
sampled_df = df[df['total_schedule_round'].isin(sampled_rounds)]

# Drop the 'schedule_round' column
sampled_df = sampled_df.drop(columns=['schedule_round'])

# Apply z-score normalization to all columns except 'total_schedule_round'
cols_to_zscore = [col for col in sampled_df.columns if col != 'total_schedule_round']
mean = sampled_df[cols_to_zscore].mean()
std = sampled_df[cols_to_zscore].std()
sampled_df[cols_to_zscore] = (sampled_df[cols_to_zscore] - mean) / std

# Apply min-max normalization to all columns except 'total_schedule_round'
# cols_to_normalize = [col for col in sampled_df.columns if col != 'total_schedule_round']
# min_vals = sampled_df[cols_to_normalize].min()
# max_vals = sampled_df[cols_to_normalize].max()
# sampled_df[cols_to_normalize] = (sampled_df[cols_to_normalize] - min_vals) / (max_vals - min_vals)
# mean = min_vals  # For saving normalization params, use min as "mean"
# std = max_vals   # For saving normalization params, use max as "std"

# Apply softmax to 'weight' values within each 'total_schedule_round'
# sampled_df['weight'] = sampled_df.groupby('total_schedule_round')['weight'].transform(softmax)

# Save the normalization parameters
torch.save({'mean': mean, 'std': std}, 'normalization_params_zscore.pt')

# Save the processed DataFrame to a new CSV file
sampled_df.to_csv('/home/maximilianrosca/Masterarbeit/Simulation-files/training_data_zscore.csv', index=False)
