# Pulsar Classifier

Identifying pulsar candidates from radio telescope data using logistic regression.

## Dataset

HTRU2 dataset from the Parkes Observatory in Australia.

- Total candidates: 17,898
- Real pulsars: 1,639 (9%)
- Non-pulsars: 16,259 (91%)
- Features: 8 (4 from integrated pulse profile, 4 from DM-SNR curve)

## Training Process

1. Loaded the data and added column names.
2. Separated features (X) and target (y).
3. Split into training (80%) and testing (20%) sets.
4. Scaled features using RobustScaler (because DM_kurtosis had extreme outliers).
5. Trained a logistic regression model with class_weight='balanced' to handle imbalance.
6. Evaluated on the test set using recall, precision, confusion matrix, and ROC-AUC.

## Results

| Metric | Score |
|--------|-------|
| Recall (Pulsars) | 92% |
| Precision | 77% |
| ROC-AUC | 0.974 |
| True Positives | 294 |
| False Positives | 86 |
| False Negatives | 27 |

Confusion Matrix:

[[3173, 86],
 [27, 294]]

## Key Decisions

- **RobustScaler over StandardScaler** because DM_kurtosis had extreme outliers (max 1191 vs 99th percentile 517).
- **class_weight='balanced'** so the model didn't ignore the rare pulsar class.
- **Recall over accuracy** because with 91% noise, accuracy is meaningless.

## Model Confidence

The model outputs a probability score. For example: 99.97% (high confidence pulsar), 85.3% (likely), 12.1% (probably noise).

## Repository Files

- HTRU_2.csv – raw dataset
- pulsar_classifier.ipynb – full training notebook
- pulsar_classifier.pkl – trained model
- robust_scaler.pkl – fitted scaler

## Requirements

Python 3.7+ with pandas, numpy, scikit-learn, matplotlib, seaborn, joblib.

References
R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656

R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1

Author
Shreyas

## Usage Example

```python
import joblib
model = joblib.load('pulsar_classifier.pkl')
scaler = joblib.load('robust_scaler.pkl')
features = [Profile_mean, Profile_stdev, Profile_skewness, Profile_kurtosis,
            DM_mean, DM_stdev, DM_skewness, DM_kurtosis]


text
features_scaled = scaler.transform([features])
pred = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0][1]
