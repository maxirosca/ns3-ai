import pandas as pd

data = pd.read_csv('/home/maximilianrosca/ns-3-dev/dl_weights_log.csv')

# Remove rows with NaN values
data = data.dropna()

# Remove rows where 'avg_tput' or 'potential_tput' are less than or equal to 0
data = data[(data['avg_tput'] > 0) & (data['potential_tput'] > 0)]

# Remove rows where weight is suspiciously high (outliers)
data = data[data['weight'] < 1e+6]

# Sample 100,000 rows for training
data_sample = data.sample(n=100000, random_state=42)

# Save the sample to a new CSV file
data_sample.to_csv('/home/maximilianrosca/ns-3-dev/dl_weights_log_sample.csv', index=False)