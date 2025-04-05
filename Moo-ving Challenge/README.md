# Moo-ving Insights: Eructation Prediction Challenge

This repository contains a Jupyter Notebook (`Eructation_Prediction.ipynb`) developed for a coding challenge focused on analyzing sensor data from two cows and prototyping a model to predict eructation events using thermistor data.

## Context

In this challenge, data was collected from two cows (Cow A and Cow B) over approximately one hour. Each cow was equipped with:
- **2 thermistors**: Sensors measuring breath temperature.
- **1 gas sensor**: Measuring CH4 and CO2 concentrations in exhaled breath.

The objective is to explore the dataset and build a predictive model for eructation (periodic belching of rumen gas), a key process tied to gas elimination in cows.

**Note**: The data is simulated and does not reflect real cow behavior.

## Dataset

The dataset includes:
1. **`cow[A,B]_therm-[0,2].csv`**: Thermistor data for Cow A and Cow B.
   - **Columns**:
     - `time_ms`: Epoch timestamp in milliseconds.
     - `thermistor_id`: Thermistor identifier (0 or 2).
     - `raw`: Thermistor voltage ratio.
     - `temperature`: Temperature in Celsius derived from the raw voltage ratio.

2. **`cow[A,B]_gas.csv`**: Gas sensor data for Cow A and Cow B.
   - **Columns**:
     - `epoch_ms`: Epoch timestamp in milliseconds.
     - `co2`: CO2 percent concentration.
     - `co2temp`: Temperature of gas in the sensor box (Celsius).
     - `ch4`: CH4 percent concentration.
     - `ch4temp`: Temperature of gas in the sensor box (Celsius).

## Tasks

The Jupyter Notebook tackles these tasks:
1. **Data Analysis**:
   - Clean the dataset (e.g., handle missing values, inconsistencies).
   - Explore data characteristics (e.g., distributions, time series trends).
   - Create visualizations to reveal insights.

2. **Prototype Model Development**:
   - Define the problem: Predict eructation events using thermistor data, with CH4 peaks as proxies.
   - Build an XGBoost model with time-based features to capture sequential patterns.
   - Evaluate performance with accuracy, precision, recall, and ROC AUC metrics.

## Assumptions

- **Eructation Proxy**: Peaks in CH4 concentration indicate eructation events.
- **Sampling Rates**: Gas data is sampled ~every 500ms, thermistor data ~every 40ms.
- **Time Alignment**: Datasets are merged by timestamps, assuming reasonable synchronization.

## Approach

1. **Data Preprocessing**:
   - Load and clean thermistor and gas data.
   - Merge datasets based on timestamps.
   - Label eructation events using CH4 peak thresholds.

2. **Feature Engineering**:
   - Generate lagged features and rolling window statistics (mean, std) for thermistor temperatures.
   - Normalize features with `StandardScaler`.

3. **Model Selection**:
   - Use **XGBoost** for its efficiency with tabular data and engineered time-based features.
   - Compare against a baseline (e.g., random guessing).

4. **Evaluation**:
   - Split data: 70% training, 15% validation, 15% test.
   - Report metrics: accuracy, precision, recall, and ROC AUC.

## Results

- **Validation Metrics**:
  - Accuracy: ~0.8125
  - Precision: ~0.7915
  - Recall: ~0.0387
  - ROC AUC: ~0.7762
- **Test Metrics**:
  - Accuracy: ~0.8152
  - Precision: ~0.8264
  - Recall: ~0.0415
  - ROC AUC: ~0.7744

The model slightly outperforms a baseline but struggles with low recall due to rare eructation events. Minor overfitting suggests further refinement potential.

## Next Steps

To boost performance:
1. **Synthetic Data**: Generate additional time-series data using GANs to align timestamps.
2. **Transfer Learning**: Adapt pre-trained time-series models.
3. **LSTM/GRU**: Use a small LSTM or GRU with dropout to capture temporal patterns.
4. **Hybrid Model**: Combine XGBoost with LSTM for enhanced results.
5. **Tuning**: Optimize hyperparameters to improve recall and reduce overfitting.

## Issues Encountered

- **Sampling Disparity**: Merging data with different rates (40ms vs. 500ms) is tricky.
- **Class Imbalance**: Rare eructation events lead to low recall.
- **Data Volume**: Limited to ~1 hour per cow, constraining generalization.

## Requirements

Install these Python packages to run the notebook:
```bash
pip install pandas numpy matplotlib seaborn sklearn xgboost statsmodels
