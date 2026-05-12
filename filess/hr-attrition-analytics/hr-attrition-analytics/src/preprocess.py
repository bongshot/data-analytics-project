"""Data preprocessing utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def clean_ibm(df):
    """Clean IBM dataset by dropping constants and encoding binary variables."""
    constants = ['EmployeeCount', 'StandardHours', 'Over18']
    df.drop(columns=[c for c in constants if c in df.columns], inplace=True)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

def clean_linkedin(df):
    """Filter and normalize LinkedIn dataset."""
    df = df[df['med_salary'].notnull()]
    df['title'] = df['title'].str.lower().str.strip()
    max_applies = df['applies'].max()
    df['job_demand_score'] = df['applies'] / max_applies if max_applies > 0 else 0
    return df

def merge_datasets(ibm_df, linkedin_df, title_map):
    """Merge IBM and LinkedIn data, and engineer novel features."""
    li_agg = linkedin_df.groupby('title').agg({
        'med_salary': 'median',
        'job_demand_score': 'mean',
        'remote_allowed': lambda x: (x == 1).mean()
    }).reset_index()
    
    ibm_df['Mapped_LI_Title'] = ibm_df['JobRole'].map(title_map)
    merged_df = pd.merge(ibm_df, li_agg, left_on='Mapped_LI_Title', right_on='title', how='left')
    
    merged_df['salary_gap'] = merged_df['med_salary'] - (merged_df['MonthlyIncome'] * 12)
    merged_df['external_demand_score'] = merged_df['job_demand_score'].fillna(0)
    merged_df['remote_opportunity'] = merged_df['remote_allowed'].fillna(0)
    
    merged_df.drop(columns=['Mapped_LI_Title', 'title', 'med_salary', 'job_demand_score', 'remote_allowed'], inplace=True)
    return merged_df

def encode_and_split(df, target_col='Attrition', test_size=0.2, random_state=42):
    """Encode categoricals and split data into train/test sets."""
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to training data."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)
