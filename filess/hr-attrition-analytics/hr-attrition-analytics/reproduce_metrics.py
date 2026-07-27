# Reproduce metrics with SMOTE inside CV and save artifacts
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel
import joblib

# Optional: silence warnings from shap heavy import
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

# Adjust base_dir to project internal path
base_dir = "filess/hr-attrition-analytics/hr-attrition-analytics"
proc_path = os.path.join(base_dir, "data", "processed", "merged_dataset_raw.csv")

if not os.path.exists(proc_path):
    raise FileNotFoundError(f"Processed dataset not found at {proc_path}. Please run the pipeline to generate merged_dataset_raw.csv under data/processed.")

os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

print('Loading processed dataset...')
df = pd.read_csv(proc_path)
if 'Attrition_Num' not in df.columns:
    raise KeyError('Attrition_Num column not found in processed dataset. Ensure target is encoded as Attrition_Num (0/1).')

y = df['Attrition_Num']
X = df.drop(columns=['Attrition_Num']).fillna(0)

novel_cols = [c for c in ['salary_gap', 'external_demand_score', 'remote_opportunity'] if c in X.columns]
X_ibm = X.drop(columns=novel_cols) if novel_cols else X.copy()
X_plus = X.copy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'f1': make_scorer(f1_score), 'roc_auc': 'roc_auc', 'accuracy': 'accuracy'}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

out_records = []
cv_details = {}

for name, model in models.items():
    # Build pipeline: SMOTE only in training folds
    pipe = ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', model)])
    for label, X_mat in [('IBM_ONLY', X_ibm), ('IBM_PLUS', X_plus)]:
        print(f'Running CV for {name} on {label}...')
        res = cross_validate(pipe, X_mat, y, cv=cv, scoring=scoring, return_estimator=True, n_jobs=-1)
        mean_metrics = {k: float(np.mean(v)) for k,v in res.items() if k.startswith('test_')}
        mean_metrics = {
            'Model': name,
            'Dataset': label,
            'Accuracy': mean_metrics.get('test_accuracy'),
            'F1': mean_metrics.get('test_f1'),
            'AUC': mean_metrics.get('test_roc_auc'),
        }
        out_records.append(mean_metrics)
        # Save per-fold F1
        cv_details[f"{name}__{label}"] = {
            'test_f1': [float(x) for x in res['test_f1']],
            'test_roc_auc': [float(x) for x in res['test_roc_auc']],
            'test_accuracy': [float(x) for x in res['test_accuracy']]
        }
        # Save final estimator trained on full X_plus for feature importance (for XGBoost only)
        if name == 'XGBoost' and label == 'IBM_PLUS':
            print('Training final XGBoost pipeline on full IBM_PLUS...')
            final_pipe = pipe
            final_pipe.fit(X_plus, y)
            clf = final_pipe.named_steps['clf']
            # feature importances
            importances = pd.Series(clf.feature_importances_, index=X_plus.columns).sort_values(ascending=False)
            importances.iloc[:50].to_csv(os.path.join(base_dir, 'outputs', 'xgb_feature_importances.csv'))
            joblib.dump(clf, os.path.join(base_dir, 'outputs', 'xgb_plus_model.joblib'))
            # SHAP if available
            if shap_available:
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_plus)
                    shap_mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_plus.columns).sort_values(ascending=False)
                    shap_mean_abs.head(50).to_csv(os.path.join(base_dir, 'outputs', 'shap_mean_abs.csv'))
                except Exception as e:
                    print('SHAP computation failed:', e)

# Save metrics and CV details
metrics_path = os.path.join(base_dir, 'outputs', 'model_metrics.csv')
cv_path = os.path.join(base_dir, 'outputs', 'cv_folds.json')
with open(metrics_path, 'w') as f:
    pd.DataFrame(out_records).to_csv(f, index=False)
with open(cv_path, 'w') as f:
    json.dump(cv_details, f, indent=2)

# Paired t-test for XGBoost F1 arrays
f1_only = np.array(cv_details['XGBoost__IBM_ONLY']['test_f1'])
f1_plus = np.array(cv_details['XGBoost__IBM_PLUS']['test_f1'])
from scipy.stats import ttest_rel
if len(f1_only) != len(f1_plus):
    raise ValueError('Fold counts differ between IBM_ONLY and IBM_PLUS')

t_stat, p_val = ttest_rel(f1_plus, f1_only)
tt = {'t_stat': float(t_stat), 'p_value': float(p_val), 'mean_delta': float(f1_plus.mean() - f1_only.mean()), 'df': int(len(f1_plus)-1)}
with open(os.path.join(base_dir, 'outputs', 'ttest_results.json'), 'w') as f:
    json.dump(tt, f, indent=2)

print('Artifacts written to:', os.path.join(base_dir, 'outputs'))
