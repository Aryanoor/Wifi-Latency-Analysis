# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# -----------------------------
# Step 1: Load and Preprocess Dataset
# -----------------------------

# Load dataset from CSV
df = pd.read_csv('wifi_latency.csv')

# Select only the relevant columns for analysis
important_columns = ['latency_ms', 'rssi_dbm', 'snr_db', 'channel_util%',
                     'num_assoc_devices', 'client_speed_mbps', 'distance_m',
                     'band', 'protocol', 'ap_vendor']
df = df[important_columns]

# Drop any rows containing missing values (NaN)
df = df.dropna()

# Define list of numeric features for outlier filtering
numeric_features = ['latency_ms', 'rssi_dbm', 'snr_db', 'channel_util%',
                    'num_assoc_devices', 'client_speed_mbps', 'distance_m']

# Remove outliers using the IQR method for each numeric feature
for col in numeric_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Filter out logically invalid values (e.g., negative distances or out-of-bound percentages)
df = df[df['distance_m'] >= 0]
df = df[df['client_speed_mbps'] >= 0]
df = df[df['channel_util%'].between(0, 100)]
df = df[df['snr_db'] >= 0]

# Convert categorical columns to 'category' type for optimized storage and performance
df['band'] = df['band'].astype('category')
df['protocol'] = df['protocol'].astype('category')
df['ap_vendor'] = df['ap_vendor'].astype('category')

# -----------------------------
# Frame 1: Boxplot for Categorical Features
# -----------------------------

# Define categorical features to plot
categorical_features = ['band', 'protocol', 'ap_vendor']

# Create subplots: one for each categorical feature
fig1, axes1 = plt.subplots(1, len(categorical_features), figsize=(18, 5))
fig1.suptitle('Frame 1: Latency Distribution by Categorical Features (Boxplot)', fontsize=16)

# Generate boxplot for each categorical feature against latency
for i, feature in enumerate(categorical_features):
    sns.boxplot(x=feature, y='latency_ms', data=df, ax=axes1[i], palette='Set2')
    axes1[i].set_title(feature)
    axes1[i].set_ylabel('Latency (ms)')
    axes1[i].tick_params(axis='x', rotation=30)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()

# -----------------------------
# Frame 2: Trend Plot for Numeric Features
# -----------------------------

# Create a 2x3 grid of plots for numeric feature analysis
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10))
fig2.suptitle('Frame 2: Latency Trend by Numeric Features (Binned)', fontsize=16)

# Plot mean latency trend across 10 bins for each numeric feature (excluding latency itself)
for i, feature in enumerate(numeric_features[1:]):  # skip 'latency_ms'
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

# -----------------------------
# Frame 3: Correlation Matrix
# -----------------------------

# Copy dataset for encoding categorical features
df_encoded = df.copy()

# Encode each categorical feature into integer values
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Compute correlation matrix on numeric data (encoded + original numeric)
corr_matrix = df_encoded.corr(numeric_only=True)

# Visualize correlation matrix using heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Frame 3: Correlation Matrix Including Encoded Categorical Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# Frame 4: Mutual Information (Manual Implementation)
# -----------------------------

# Copy data for MI calculation
df_mi = df.copy()

# Create log-transformed latency and divide into 6 quantile-based bins
df_mi['latency_binned'] = pd.qcut(np.log1p(df_mi['latency_ms']), q=6, duplicates='drop')

# Fill any missing values in 'distance_m' if they exist
if df_mi['distance_m'].isnull().sum() > 0:
    df_mi['distance_m'].fillna(15, inplace=True)

# Bin numeric features into 6 equal-width bins using KBinsDiscretizer
binned_features = {}
for col in ['rssi_dbm', 'snr_db', 'channel_util%', 'num_assoc_devices', 
            'client_speed_mbps', 'distance_m']:
    kbinner = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
    df_mi[col + '_binned'] = kbinner.fit_transform(df_mi[[col]]).astype(int)
    binned_features[col] = col + '_binned'

# Encode categorical features into numeric form
for col in categorical_features:
    le = LabelEncoder()
    df_mi[col + '_encoded'] = le.fit_transform(df_mi[col])

# Combine all features to be analyzed for MI
all_features = list(binned_features.values()) + [f + '_encoded' for f in categorical_features]

# Define mutual information computation function
def compute_mutual_information(x, y):
    # Create a contingency table
    joint_df = pd.crosstab(x, y)

    # Convert to joint probability table
    joint_prob = joint_df / joint_df.values.sum()

    # Marginal probabilities for x and y
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    # Mutual Information computation based on MI formula
    mi = 0.0
    for xi in joint_df.index:
        for yi in joint_df.columns:
            pxy = joint_prob.loc[xi, yi]
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px[xi] * py[yi]))
    return mi

# Compute MI score between each feature and binned latency
mi_scores = {feature: compute_mutual_information(df_mi[feature], df_mi['latency_binned']) for feature in all_features}

# Create DataFrame for MI results and sort them
mi_df = pd.DataFrame.from_dict(mi_scores, orient='index', columns=['MI_with_latency'])
mi_df = mi_df.sort_values(by='MI_with_latency', ascending=False)

# Plot MI scores as bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=mi_df['MI_with_latency'], y=mi_df.index, palette='viridis')
plt.title('Frame 4: Mutual Information between Features and Latency (Manual Calculation)')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
