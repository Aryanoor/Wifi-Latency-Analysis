# -----------------------------------------------------
#                  IMPORT LIBRARIES
# -----------------------------------------------------
"""
Import all necessary Python libraries:

- pandas: for data manipulation and loading
- numpy: for numerical computations
- seaborn: for statistical data visualization
- matplotlib: for general plotting
- sklearn.preprocessing:
    - LabelEncoder: to encode categorical values into integers
    - KBinsDiscretizer: to bin numeric features for mutual information analysis
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# -----------------------------------------------------
#                  LOAD DATA
# -----------------------------------------------------
"""
Load the dataset from CSV into a pandas DataFrame.
"""
df = pd.read_csv('wifi_latency.csv')

# -----------------------------------------------------
#              DATA PRE-PROCESSING
# -----------------------------------------------------
"""
Data Cleaning and Preparation Steps:
1. Retain only relevant columns.
2. Remove missing values (NaNs).
3. Remove statistical outliers using IQR.
4. Apply domain-based logical filters.
5. Convert categorical columns to 'category' dtype for efficiency.
"""

# Select relevant features only
important_columns = ['latency_ms', 'rssi_dbm', 'snr_db', 'channel_util%',
                     'num_assoc_devices', 'client_speed_mbps', 'distance_m',
                     'band', 'protocol', 'ap_vendor']
df = df[important_columns]

# Drop rows with missing values to ensure clean dataset
df = df.dropna()

# Define numeric features to clean
numeric_features = ['latency_ms', 'rssi_dbm', 'snr_db', 'channel_util%',
                    'num_assoc_devices', 'client_speed_mbps', 'distance_m']

# Remove outliers using IQR method for each numeric feature
for col in numeric_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Logical data filters to maintain valid ranges
df = df[df['distance_m'] >= 0]
df = df[df['client_speed_mbps'] >= 0]
df = df[df['channel_util%'].between(0, 100)]
df = df[df['snr_db'] >= 0]

# Convert categorical features to 'category' data type
df['band'] = df['band'].astype('category')
df['protocol'] = df['protocol'].astype('category')
df['ap_vendor'] = df['ap_vendor'].astype('category')

# -----------------------------------------------------
#      CATEGORICAL FEATURE ANALYSIS (BOXPLOTS)
# -----------------------------------------------------
"""
Boxplots are used to visualize the distribution of latency across different
categories such as:
- Band (e.g., 2.4GHz, 5GHz)
- Protocol (e.g., 802.11n, 802.11ac)
- Access Point Vendor

These plots help in identifying performance differences between categories.
"""

categorical_features = ['band', 'protocol', 'ap_vendor']
fig1, axes1 = plt.subplots(1, len(categorical_features), figsize=(18, 5))
fig1.suptitle('Latency Distribution by Categorical Features (Boxplot)', fontsize=16)

for i, feature in enumerate(categorical_features):
    sns.boxplot(x=feature, y='latency_ms', data=df, ax=axes1[i], palette='Set2')
    axes1[i].set_title(feature)
    axes1[i].set_ylabel('Latency (ms)')
    axes1[i].tick_params(axis='x', rotation=30)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()

# -----------------------------------------------------
#    NUMERIC FEATURE ANALYSIS (BINNED TREND PLOTS)
# -----------------------------------------------------
"""
For numeric features, we bin them into 10 equal-width intervals, then plot
average latency in each bin. This reveals trends such as:
- How signal strength or distance affects latency
- Relationship between utilization or device count and latency
"""

fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10))
fig2.suptitle('Latency Trend by Numeric Features (Binned)', fontsize=16)

for i, feature in enumerate(numeric_features[1:]):  # Exclude latency itself
    df['bin'] = pd.cut(df[feature], bins=10)
    grouped = df.groupby('bin', observed=False)['latency_ms'].mean().reset_index()
    row = i // 3
    col = i % 3
    ax = axes2[row, col]
    ax.plot(grouped['bin'].astype(str), grouped['latency_ms'], marker='o')
    ax.set_title(f'{feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Avg Latency (ms)')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()

# -----------------------------------------------------
#         CORRELATION MATRIX (ENCODED DATA)
# -----------------------------------------------------
"""
Compute and visualize Pearson correlation between numeric and encoded categorical features.
Encoding categorical features as integers enables inclusion in the correlation matrix.
"""

df_encoded = df.copy()
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

corr_matrix = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Including Encoded Categorical Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
#      HISTOGRAM OF LOG-TRANSFORMED LATENCY
# -----------------------------------------------------
"""
Log-transform latency using log1p to reduce skewness and better visualize its distribution.
Histogram shows smoothed (KDE) and normalized view of latency values.
"""

plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df['latency_ms']), bins=30, kde=True, stat='density', color='steelblue')
plt.title('Normalized Histogram of log(latency_ms)')
plt.xlabel('log(1 + latency_ms)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# -----------------------------------------------------
#                  MUTUAL INFORMATION
# -----------------------------------------------------
"""
Mutual Information (MI) measures the dependency between input features and
log-transformed, binned latency.

Steps:
1. Bin numerical features using KBinsDiscretizer.
2. Encode categorical features.
3. Bin latency into quantiles.
4. Calculate MI for each feature relative to latency.
"""

df_mi = df.copy()
df_mi['latency_binned'] = pd.qcut(np.log1p(df_mi['latency_ms']), q=6, duplicates='drop')

# Fill any missing distances (as a precaution)
if df_mi['distance_m'].isnull().sum() > 0:
    df_mi['distance_m'].fillna(15, inplace=True)

# Bin numeric features uniformly
binned_features = {}
for col in ['rssi_dbm', 'snr_db', 'channel_util%', 'num_assoc_devices',
            'client_speed_mbps', 'distance_m']:
    kbinner = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
    df_mi[col + '_binned'] = kbinner.fit_transform(df_mi[[col]]).astype(int)
    binned_features[col] = col + '_binned'

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df_mi[col + '_encoded'] = le.fit_transform(df_mi[col])

# Combine all transformed features
all_features = list(binned_features.values()) + [f + '_encoded' for f in categorical_features]

# Function to compute mutual information
def compute_mutual_information(x, y):
    joint_df = pd.crosstab(x, y)
    joint_prob = joint_df / joint_df.values.sum()
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)
    mi = 0.0
    for xi in joint_df.index:
        for yi in joint_df.columns:
            pxy = joint_prob.loc[xi, yi]
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px[xi] * py[yi]))
    return mi

# Calculate and store MI values
mi_matrix = pd.DataFrame(index=all_features, columns=['latency_binned'])

for feature in all_features:
    mi_matrix.loc[feature, 'latency_binned'] = compute_mutual_information(df_mi[feature], df_mi['latency_binned'])

mi_matrix = mi_matrix.astype(float)
mi_matrix = mi_matrix.sort_values(by='latency_binned', ascending=False)

# -----------------------------------------------------
#             PLOT MUTUAL INFORMATION HEATMAP
# -----------------------------------------------------

plt.figure(figsize=(6, len(all_features) * 0.5 + 2))
sns.heatmap(mi_matrix, annot=True, cmap='YlGnBu', fmt=".5f", linewidths=0.5)
plt.title('Mutual Information Heatmap (Features vs. Binned Latency)')
plt.xlabel('Target: latency_binned')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# -----------------------------------------------------
#             PLOT MUTUAL INFORMATION BARPLOT
# -----------------------------------------------------
"""
Alternative view: Barplot to rank features based on their mutual information
with the binned latency.
"""

mi_scores = {feature: compute_mutual_information(df_mi[feature], df_mi['latency_binned']) for feature in all_features}
mi_df = pd.DataFrame.from_dict(mi_scores, orient='index', columns=['MI_with_latency'])
mi_df = mi_df.sort_values(by='MI_with_latency', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_df['MI_with_latency'], y=mi_df.index, palette='viridis')
plt.title('Mutual Information between Features and Latency (Manual Calculation)')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()