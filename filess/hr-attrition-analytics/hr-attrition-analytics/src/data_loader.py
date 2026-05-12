"""Data loading utilities."""
import pandas as pd

def load_ibm(path):
    """Load IBM HR dataset and print summary."""
    df = pd.read_csv(path)
    print(f"IBM Data loaded. Shape: {df.shape}")
    print(df['Attrition'].value_counts(normalize=True))
    return df

def load_linkedin(path):
    """Load LinkedIn Job dataset and print summary."""
    df = pd.read_csv(path)
    print(f"LinkedIn Data loaded. Shape: {df.shape}")
    print(f"Salary completeness: {df['med_salary'].notnull().mean():.2%}")
    return df

def get_job_title_map():
    """Return IBM JobRole to LinkedIn title mapping."""
    return {
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
