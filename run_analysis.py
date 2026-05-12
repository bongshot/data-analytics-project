import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
base_dir = "e:/vscode/data analytics project/filess/hr-attrition-analytics/hr-attrition-analytics"
os.makedirs(f"{base_dir}/reports/figures", exist_ok=True)
os.makedirs(f"{base_dir}/data/processed", exist_ok=True)

# 1. LOAD DATA
print("Loading data...")
ibm_path = f"{base_dir}/data/raw/ibm_hr_attrition.csv"
linkedin_path = f"{base_dir}/data/raw/linkedin_jobs.csv"

ibm_df = pd.read_csv(ibm_path)
li_df = pd.read_csv(linkedin_path)

# EDA: IBM
print("Running EDA...")
# Fig 1
plt.figure(figsize=(8,5))
plt_sns.countplot(x='Attrition', data=ibm_df, palette=['#2ECC71', '#E74C3C'])
plt.title("Attrition Class Balance")
plt.savefig(f"{base_dir}/reports/figures/fig01_attrition_balance.png", bbox_inches='tight')

# Encode target for correlation
ibm_df['Attrition_Num'] = ibm_df['Attrition'].map({'Yes': 1, 'No': 0})
numeric_cols = ibm_df.select_dtypes(include=[np.number]).columns

# Fig 2
plt.figure(figsize=(12,10))
corr = ibm_df[numeric_cols].corr()
plt_sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.savefig(f"{base_dir}/reports/figures/fig02_correlation_heatmap.png", bbox_inches='tight')

top_corr = corr['Attrition_Num'].abs().sort_values(ascending=False)[1:11]
print("Top 10 Correlations:\n", top_corr)

# Fig 3
plt.figure(figsize=(8,5))
plt_sns.boxplot(x='Attrition', y='MonthlyIncome', data=ibm_df, palette=['#2ECC71', '#E74C3C'])
plt.savefig(f"{base_dir}/reports/figures/fig03_income_attrition.png", bbox_inches='tight')

# Preprocessing
# Drop constants
constants = ['EmployeeCount', 'StandardHours', 'Over18']
ibm_df.drop(columns=[c for c in constants if c in ibm_df.columns], inplace=True)
ibm_df['OverTime_Num'] = ibm_df['OverTime'].map({'Yes': 1, 'No': 0})
ibm_df['Gender_Num'] = ibm_df['Gender'].map({'Male': 1, 'Female': 0})

# LinkedIn Clean
li_df['title'] = li_df['title'].str.lower().str.strip()
if li_df['applies'].max() > 0:
    li_df['job_demand_score'] = li_df['applies'] / li_df['applies'].max()
else:
    li_df['job_demand_score'] = 0

# Mapping
title_map = {
    'Healthcare Representative': 'healthcare representative',
    'Human Resources': 'human resources',
    'Laboratory Technician': 'laboratory technician',
    'Manager': 'manager',
    'Manufacturing Director': 'manufacturing director',
    'Research Director': 'research director',
    'Research Scientist': 'research scientist',
    'Sales Executive': 'sales executive',
    'Sales Representative': 'sales representative'
}

# Aggregate LI data by title
li_agg = li_df.groupby('title').agg({
    'med_salary': 'median',
    'job_demand_score': 'mean',
    'remote_allowed': lambda x: (x == 1).mean()
}).reset_index()

ibm_df['Mapped_LI_Title'] = ibm_df['JobRole'].map(title_map)
merged_df = pd.merge(ibm_df, li_agg, left_on='Mapped_LI_Title', right_on='title', how='left')

# Novel Features
merged_df['salary_gap'] = merged_df['med_salary'] - (merged_df['MonthlyIncome'] * 12)
merged_df['external_demand_score'] = merged_df['job_demand_score'].fillna(0)
merged_df['remote_opportunity'] = merged_df['remote_allowed'].fillna(0)

# Drop intermediate columns
merged_df.drop(columns=['Mapped_LI_Title', 'title', 'med_salary', 'job_demand_score', 'remote_allowed', 'Attrition'], inplace=True)

# Encode remaining categoricals
cat_cols = merged_df.select_dtypes(include=['object']).columns
merged_df = pd.get_dummies(merged_df, columns=cat_cols, drop_first=True)

# Save RAW
merged_df.to_csv(f"{base_dir}/data/processed/merged_dataset_raw.csv", index=False)

y = merged_df['Attrition_Num']
X = merged_df.drop(columns=['Attrition_Num'])

# Fill NaNs with 0 (or a suitable default) to avoid SMOTE errors
X = X.fillna(0)

# Split sets
novel_cols = ['salary_gap', 'external_demand_score', 'remote_opportunity']
X_ibm = X.drop(columns=novel_cols)
X_plus = X.copy()

X_train_ibm, X_test_ibm, y_train, y_test = train_test_split(X_ibm, y, test_size=0.2, random_state=42, stratify=y)
X_train_plus, X_test_plus, _, _ = train_test_split(X_plus, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_ibm_sm, y_train_sm = smote.fit_resample(X_train_ibm, y_train)
X_train_plus_sm, _ = smote.fit_resample(X_train_plus, y_train)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=5, random_state=42)
}

results = []
roc_curves = {}

print("Training models...")
for name, model in models.items():
    # IBM Only
    model.fit(X_train_ibm_sm, y_train_sm)
    y_pred = model.predict(X_test_ibm)
    y_proba = model.predict_proba(X_test_ibm)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    res_ibm = {
        'Model': name, 'Dataset': 'IBM_ONLY',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }
    results.append(res_ibm)
    
    # IBM Plus
    model.fit(X_train_plus_sm, y_train_sm)
    y_pred_p = model.predict(X_test_plus)
    y_proba_p = model.predict_proba(X_test_plus)[:, 1] if hasattr(model, 'predict_proba') else y_pred_p
    res_plus = {
        'Model': name, 'Dataset': 'IBM_PLUS',
        'Accuracy': accuracy_score(y_test, y_pred_p),
        'F1': f1_score(y_test, y_pred_p),
        'AUC': roc_auc_score(y_test, y_proba_p)
    }
    results.append(res_plus)
    roc_curves[f"{name} (IBM+LinkedIn)"] = roc_curve(y_test, y_proba_p)
    
    if name == 'XGBoost':
        xgb_plus_model = model

df_res = pd.DataFrame(results)
print(df_res)

# Feature Importance
plt.figure(figsize=(10,8))
importances = xgb_plus_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
colors = ['red' if X_train_plus.columns[i] in novel_cols else 'steelblue' for i in indices]

plt.barh(range(20), importances[indices][::-1], color=colors[::-1])
plt.yticks(range(20), [X_train_plus.columns[i] for i in indices][::-1])
plt.title("Top 20 Features (XGBoost IBM+LinkedIn)")
plt.savefig(f"{base_dir}/reports/figures/fig12_feature_importance.png", bbox_inches='tight')

# T-test for XGBoost cross_val_score
cv_ibm = cross_val_score(models['XGBoost'], X_ibm, y, cv=5, scoring='f1')
cv_plus = cross_val_score(models['XGBoost'], X_plus, y, cv=5, scoring='f1')
t_stat, p_val = ttest_rel(cv_ibm, cv_plus)
print(f"Paired T-test p-value: {p_val}")

print("Analysis complete. Check reports/figures.")
