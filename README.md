# Wi-Fi Latency Analysis

## Overview

This project focuses on analyzing and identifying the key factors that impact latency in Wi-Fi networks. Latency, measured in milliseconds, plays a critical role in the user experience for real-time applications such as video calls, online gaming, and streaming. The analysis uses real-world data collected from various Wi-Fi client connections to explore correlations, trends, and dependencies using statistical techniques and information theory.

## Objectives

- Clean and preprocess Wi-Fi performance data
- Visualize latency distribution across categorical and numerical features
- Identify linear relationships via correlation analysis
- Transform and bin data for structured interpretation
- Compute and rank mutual information to quantify feature importance
- Provide actionable insights for network optimization

## Dataset Description

The dataset used in this project includes measurements from client connections to Wi-Fi access points. Each row represents a unique connection snapshot. Key features include:

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `latency_ms`             | Measured network latency in milliseconds        |
| `rssi_dbm`               | Received signal strength in dBm                 |
| `snr_db`                 | Signal-to-noise ratio in dB                     |
| `channel_util%`          | Channel utilization percentage                  |
| `num_assoc_devices`      | Number of devices connected to the AP           |
| `client_speed_mbps`      | Reported client connection speed in Mbps        |
| `distance_m`             | Estimated distance between client and AP        |
| `band`                   | Frequency band (e.g., 2.4GHz or 5GHz)           |
| `protocol`               | Wi-Fi protocol (e.g., 802.11n, 802.11ac)        |
| `ap_vendor`              | Access point vendor                             |

## Project Structure

- **Preprocessing**: Data cleaning, outlier removal (IQR method), and logical filtering
- **Categorical Analysis**: Boxplots showing latency distribution per category (band, protocol, vendor)
- **Numerical Analysis**: Trend plots showing latency vs. binned numerical features
- **Correlation Matrix**: Pearson correlation between numeric and encoded categorical features
- **Log Transformation**: Histogram of `log(1 + latency_ms)` to normalize distribution
- **Mutual Information**: Manual computation of MI scores to assess feature relevance

## Key Techniques Used

- **IQR-based Outlier Removal**: Filters extreme values beyond 1.5 * IQR
- **KBinsDiscretizer**: Discretizes continuous features into uniform bins
- **Label Encoding**: Transforms categorical variables into integers
- **Mutual Information**: Quantifies nonlinear dependencies between features and target
- **Seaborn & Matplotlib**: Used extensively for visual exploration

## Results Summary

- **Top Features Impacting Latency**:
  - `snr_db`
  - `distance_m`
  - `rssi_dbm`
- Features like `channel_util%` and `num_assoc_devices` also show moderate influence.
- Categorical features such as `band` and `protocol` show limited predictive power.

Mutual Information heatmaps and bar charts confirmed that physical signal characteristics are the strongest predictors of latency.

## Requirements

This project requires the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository or download the code.
2. Ensure the `wifi_latency.csv` dataset is placed in the same directory.
3. Run the Python script in your preferred environment (Jupyter Notebook or terminal).

## Applications

- Diagnosing network performance issues
- Building machine learning models for latency prediction
- Optimizing Wi-Fi infrastructure (AP placement, protocol choice, etc.)
- Enhancing Quality of Service (QoS) monitoring systems


## Contact

For questions, feedback, or contributions, please open an issue or contact the project maintainer.
