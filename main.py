import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer, recall_score, precision_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import SMOTE for handling imbalanced datasets
from imblearn.over_sampling import SMOTE
from collections import Counter # To show class distribution

# Import XGBoost
from xgboost import XGBClassifier

# --- 1. Load the dataset ---
file_path = 'claim_data.csv' # Make sure this file is in the same directory as your script/notebook
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"Shape of the dataset: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    exit() # Exit if the file isn't found

# --- 2. Initial Data Inspection ---
print("\nColumn information and data types:")
df.info()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# --- 3. Feature Engineering and Preprocessing ---

# Convert 'Date of Service' to datetime objects for temporal features
df['Date of Service'] = pd.to_datetime(df['Date of Service'])

# Store the original index to re-align features later if needed
original_index = df.index.copy()

# --- Define 'is_fraud' target variable based on more complex rules (simulated for demo) ---
df['is_fraud'] = 0 # Default to not fraud

# Rule 1: Claims denied for 'Duplicate claim' or 'Incorrect billing'
df.loc[df['Reason Code'].isin(['Duplicate claim', 'Incorrect billing']), 'is_fraud'] = 1

# Rule 2: Claims where Billed Amount is significantly higher than Allowed Amount (e.g., > 2x)
# This could indicate upcoding or inflated billing.
df.loc[df['Billed Amount'] > (2 * df['Allowed Amount']), 'is_fraud'] = 1

# Rule 3: Claims from providers with high frequency of specific high-cost procedures (potential phantom billing)
# This is an aggregation feature.
# First, identify high-cost procedure codes (e.g., top 10% based on average billed amount)
top_n_percent_procedures = df.groupby('Procedure Code')['Billed Amount'].mean().nlargest(int(len(df['Procedure Code'].unique()) * 0.1)).index
# Then, calculate provider's frequency for these
provider_high_cost_proc_count = df[df['Procedure Code'].isin(top_n_percent_procedures)].groupby('Provider ID')['Claim ID'].count()
# Flag providers who have an unusually high count (e.g., top 5% of providers for these procedures)
if not provider_high_cost_proc_count.empty:
    threshold_high_cost_providers = provider_high_cost_proc_count.quantile(0.95)
    suspicious_providers = provider_high_cost_proc_count[provider_high_cost_proc_count > threshold_high_cost_providers].index
    df.loc[df['Provider ID'].isin(suspicious_providers), 'is_fraud'] = 1

# Rule 4: Claims where Paid Amount is 0 despite a significant Billed Amount (and not explicitly denied for a valid reason)
# This might indicate claims that were outright rejected due to very suspicious activity,
# or for services that should never have been billed.
df.loc[(df['Paid Amount'] == 0) & (df['Billed Amount'] > 50) & (~df['Reason Code'].isin(['Authorization not', 'Patient eligibility'])), 'is_fraud'] = 1


# --- NEW FEATURES ---

# Feature 1: Direct Indicator for Zero Paid Amount Denials
df['is_zero_paid_denial'] = ((df['Paid Amount'] == 0) & (df['Outcome'] == 'Denied')).astype(int)

# Feature 2: Interaction Feature: Billed Amount x (is_zero_paid_claim_denied)
df['billed_amt_if_zero_paid_denial'] = df['Billed Amount'] * df['is_zero_paid_denial']

# Feature 4: Interaction Feature: Billed Amount x (Reason Code is Patient Eligibility/Pre-existing)
df['is_patient_elig_preexist_reason'] = df['Reason Code'].isin(['Patient eligibility', 'Pre-existing cond']).astype(int)
df['billed_amt_if_patient_elig_preexist'] = df['Billed Amount'] * df['is_patient_elig_preexist_reason']

# Feature 5: Historical Provider-Patient Claims Counts (in last 365 days)
# Sort by Patient ID, Provider ID, and Date of Service for correct temporal calculation
df_temp_sorted_for_history = df.sort_values(by=['Patient ID', 'Provider ID', 'Date of Service']).copy()

historical_claims_counts = df_temp_sorted_for_history.set_index('Date of Service') \
                                                     .groupby(['Patient ID', 'Provider ID'])['Claim ID'] \
                                                     .rolling('365D', closed='left', min_periods=1).count()

historical_claims_counts_df = historical_claims_counts.reset_index()
historical_claims_counts_df.columns = ['Patient ID', 'Provider ID', 'Date of Service', 'patient_provider_claims_365d_calculated']

df = pd.merge(df, historical_claims_counts_df, on=['Patient ID', 'Provider ID', 'Date of Service'], how='left')
df['patient_provider_claims_365d'] = df['patient_provider_claims_365d_calculated'].fillna(0)
df.drop(columns=['patient_provider_claims_365d_calculated'], inplace=True)


# --- Existing Features ---

# Feature: Billed to Allowed Ratio
df['billed_to_allowed_ratio'] = df['Billed Amount'] / (df['Allowed Amount'] + 1e-6)

# Feature: Allowed to Paid Ratio
df['allowed_to_paid_ratio'] = df['Allowed Amount'] / (df['Paid Amount'] + 1e-6)

# Feature: Days since last claim by Patient
df_sorted_patient = df.sort_values(by=['Patient ID', 'Date of Service']).copy()
df['days_since_last_patient_claim'] = df_sorted_patient.groupby('Patient ID')['Date of Service'].diff().dt.days.fillna(0)


# Feature: Number of claims by Provider in last 30 days
temp_df_provider_rolling = df[['Provider ID', 'Date of Service', 'Claim ID']].copy()
temp_df_provider_rolling = temp_df_provider_rolling.set_index('Date of Service').sort_index()

provider_rolling_counts = temp_df_provider_rolling.groupby('Provider ID')['Claim ID'].rolling('30D', closed='left', min_periods=0).count()

provider_rolling_counts_df = provider_rolling_counts.reset_index()
provider_rolling_counts_df.columns = ['Provider ID', 'Date of Service', 'provider_claims_30d_calculated']

df = pd.merge(df, provider_rolling_counts_df, on=['Provider ID', 'Date of Service'], how='left')
df['provider_claims_30d'] = df['provider_claims_30d_calculated'].fillna(0)
df.drop(columns=['provider_claims_30d_calculated'], inplace=True)


# Feature: Average Billed Amount per Procedure Code by Provider (Aggregation)
df['avg_billed_per_proc_by_provider'] = df.groupby(['Provider ID', 'Procedure Code'])['Billed Amount'].transform('mean')

# Feature: Deviation from average billed amount for that procedure by that provider
df['billed_amt_dev_from_avg_proc_by_provider'] = df['Billed Amount'] - df['avg_billed_per_proc_by_provider']


# Drop original identifier columns and target leakage columns, and intermediate binary features
columns_to_drop = [
    'Claim ID', 'Provider ID', 'Patient ID',
    'Date of Service',
    'Reason Code', 'Claim Status', 'Outcome', 'Follow-up Required', 'AR Status',
    'avg_billed_per_proc_by_provider',
    'is_zero_paid_denial',
    'is_patient_elig_preexist_reason'
]
df_processed = df.drop(columns=columns_to_drop)


# Handle categorical features using One-Hot Encoding
categorical_cols = df_processed.select_dtypes(include='object').columns
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df_processed.drop('is_fraud', axis=1)
y = df_processed['is_fraud']

# Scale numerical features (important for many ML algorithms)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


print("\nProcessed data head (features):")
print(X.head())
print(f"Shape of processed features: {X.shape}")
print("\nFraudulent claims distribution (after new rules):")
print(y.value_counts())
print(f"Percentage of fraudulent claims: {y.mean()*100:.2f}%")


# --- 4. Model Training & Hyperparameter Tuning ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- Address Class Imbalance with SMOTE ---
print(f"\nOriginal training set class distribution: {Counter(y_train)}")

smote = SMOTE(random_state=42, sampling_strategy=0.7)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled training set class distribution: {Counter(y_train_resampled)}")

# --- Hyperparameter Tuning with GridSearchCV ---
print("\n--- Starting GridSearchCV for XGBoost (Optimizing for F1-score) ---")

# Calculate scale_pos_weight for the *original* imbalanced training set
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'colsample_bytree': [0.7, 0.9],
    'subsample': [0.7, 0.9],
    'gamma': [0, 0.1]
}

# Define a custom scorer for F1-score of the positive class
scorer = make_scorer(f1_score, pos_label=1)

# Initialize XGBClassifier with fixed parameters for GridSearch
xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False, # Suppress warning
    eval_metric='logloss', # Suppress warning
    scale_pos_weight=scale_pos_weight_value
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print("\n--- GridSearchCV Complete ---")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1-score found: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_
print("\nModel training complete using best parameters from GridSearchCV!")

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting the Confusion Matrix for better visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
            yticklabels=['Actual Not Fraud', 'Actual Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_prob)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")


# --- Precision-Recall Analysis and Threshold Adjustment ---
print("\n--- Precision-Recall Analysis & Threshold Adjustment ---")

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that maximizes F1-score on the test set
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6) # Add epsilon to avoid division by zero
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]
optimal_precision = precisions[optimal_threshold_idx]
optimal_recall = recalls[optimal_threshold_idx]
optimal_f1 = f1_scores[optimal_threshold_idx]

print(f"\nOptimal Threshold for F1-score: {optimal_threshold:.4f}")
print(f"Precision at optimal threshold: {optimal_precision:.4f}")
print(f"Recall at optimal threshold: {optimal_recall:.4f}")
print(f"F1-score at optimal threshold: {optimal_f1:.4f}")

# Plotting Precision and Recall vs. Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="best")
plt.title("Precision and Recall vs. Classification Threshold")
plt.grid(True)
plt.show()

# Plotting Precision-Recall Curve itself (already done, but reiterating importance)
plt.figure(figsize=(8, 6))
disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name="XGBoost")
disp.plot(ax=plt.gca())
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


# Re-evaluate with Optimal Threshold (for demonstration)
print(f"\n--- Re-evaluating with Optimal Threshold ({optimal_threshold:.4f}) ---")
y_pred_optimal_threshold = (y_prob >= optimal_threshold).astype(int)

print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal_threshold, target_names=['Not Fraud', 'Fraud']))

print("\nConfusion Matrix (Optimal Threshold):")
cm_optimal = confusion_matrix(y_test, y_pred_optimal_threshold)
print(cm_optimal)

# Plotting the Confusion Matrix for optimal threshold
plt.figure(figsize=(6, 4))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
            yticklabels=['Actual Not Fraud', 'Actual Fraud'])
plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.4f})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# --- Feature Importance Analysis ---
print("\n--- Feature Importance Analysis (XGBoost) ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("\nTop 20 Features by Importance:")
print(feature_importances.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances.head(20).values, y=feature_importances.head(20).index, palette='viridis')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.xlabel('Importance (F-score)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# --- 6. Identify Top Suspicious Claims (Example) ---
# Add predictions and probabilities back to the original test set for review
test_df_original_indices = y_test.index
predictions_df = df.loc[test_df_original_indices].copy()
predictions_df['predicted_is_fraud_default_threshold'] = y_pred # Predictions at default 0.5
predictions_df['predicted_is_fraud_optimal_threshold'] = y_pred_optimal_threshold # Predictions at optimal threshold
predictions_df['fraud_probability'] = y_prob

print("\n--- Top 10 Claims Predicted as Fraud with Highest Probability (using Optimal Threshold) ---")
# Filter based on optimal threshold prediction
top_fraud_predictions_optimal = predictions_df[predictions_df['predicted_is_fraud_optimal_threshold'] == 1].sort_values(by='fraud_probability', ascending=False).head(10)
print(top_fraud_predictions_optimal[['Claim ID', 'Billed Amount', 'Paid Amount', 'Reason Code', 'Outcome', 'fraud_probability', 'predicted_is_fraud_optimal_threshold', 'is_fraud']])

print("\n--- Top 10 Actual Fraud Claims Missed by Model (False Negatives - using Optimal Threshold) ---")
false_negatives_optimal_df = predictions_df[(predictions_df['is_fraud'] == 1) & (predictions_df['predicted_is_fraud_optimal_threshold'] == 0)].sort_values(by='fraud_probability', ascending=False).head(10)
print(false_negatives_optimal_df[['Claim ID', 'Billed Amount', 'Paid Amount', 'Reason Code', 'Outcome', 'fraud_probability', 'predicted_is_fraud_optimal_threshold', 'is_fraud']])

print("\n--- Top 10 Claims Predicted as Fraud but were NOT (False Positives - using Optimal Threshold) ---")
false_positives_optimal_df = predictions_df[(predictions_df['is_fraud'] == 0) & (predictions_df['predicted_is_fraud_optimal_threshold'] == 1)].sort_values(by='fraud_probability', ascending=False).head(10)
print(false_positives_optimal_df[['Claim ID', 'Billed Amount', 'Paid Amount', 'Reason Code', 'Outcome', 'fraud_probability', 'predicted_is_fraud_optimal_threshold', 'is_fraud']])