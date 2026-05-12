"""Modeling utilities."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

def get_models():
    """Return dictionary of uninitialized models with proper hyperparameters."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=5, random_state=42)
    }

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    """Train a single model and return its evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }

def run_all_experiments(X_train_ibm, X_test_ibm, X_train_plus, X_test_plus, y_train, y_test):
    """Run experiments across all models and datasets."""
    models = get_models()
    results = []
    roc_curves = {}
    
    for name, model in models.items():
        # IBM Only
        res_ibm = train_and_evaluate(model, X_train_ibm, y_train, X_test_ibm, y_test, name, 'IBM_ONLY')
        results.append(res_ibm)
        
        # IBM Plus
        model.fit(X_train_plus, y_train) # Re-fit on plus data to get ROC curve
        y_proba_p = model.predict_proba(X_test_plus)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_plus)
        res_plus = train_and_evaluate(model, X_train_plus, y_train, X_test_plus, y_test, name, 'IBM_PLUS')
        results.append(res_plus)
        roc_curves[f"{name} (IBM+LinkedIn)"] = roc_curve(y_test, y_proba_p)
        
    return pd.DataFrame(results), roc_curves

def plot_roc_curves(roc_curves_dict, save_path):
    """Plot ROC curves for the evaluated models."""
    plt.figure(figsize=(8,6))
    for name, (fpr, tpr, _) in roc_curves_dict.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (IBM+LinkedIn Models)')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')

def plot_feature_importance(model, feature_names, novel_features, save_path):
    """Plot feature importance and highlight novel features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    colors = ['red' if feature_names[i] in novel_features else 'steelblue' for i in indices]
    
    plt.figure(figsize=(10,8))
    plt.barh(range(20), importances[indices][::-1], color=colors[::-1])
    plt.yticks(range(20), [feature_names[i] for i in indices][::-1])
    plt.title('Top 20 Features (XGBoost IBM+LinkedIn)')
    plt.xlabel('Relative Importance')
    plt.savefig(save_path, bbox_inches='tight')
