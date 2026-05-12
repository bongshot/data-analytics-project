# Beyond Internal HR: Predicting Employee Attrition by Merging HR Analytics with External Job Market Signals

![Python Badge](https://img.shields.io/badge/Python-3.10%2B-blue)
![License Badge](https://img.shields.io/badge/License-MIT-green)
![Status Badge](https://img.shields.io/badge/Status-Active-success)

A novel data mining approach to HR Analytics integrating internal HR records with external LinkedIn job market data to predict employee attrition with enhanced accuracy.

---

## 📋 Table of Contents
- [Abstract](#abstract)
- [Novelty & Research Questions](#novelty--research-questions)
- [Datasets](#-datasets)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Results](#-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Abstract

Employee attrition is a critical challenge for modern organizations, costing billions annually in lost productivity and hiring expenses. Traditional predictive models rely exclusively on **internal HR metrics**. 

This project introduces a **hybrid framework** that combines:
- **IBM HR Analytics data** (internal HR records)
- **LinkedIn job market signals** (external labor market data)

to predict employee attrition more accurately. The approach validates the **"Pull-Push" theory** of turnover by introducing novel features like `salary_gap`, `external_demand_score`, and `remote_opportunity`.

---

## 🔬 Novelty & Research Questions

### Research Questions
1. Which internal HR factors most strongly predict employee attrition?
2. Do external job market signals (salary benchmarks, remote availability, job demand) independently predict attrition when controlling for internal factors?
3. Does a hybrid model (internal + external features) significantly outperform an internal-only model in attrition prediction?

### Key Innovation
Unlike previous studies limited to internal variables, this project bridges **internal HR data with external labor market data**. 

**Novel Features Introduced:**
- `salary_gap` — Difference between external median salary and internal employee salary
- `external_demand_score` — Normalized job postings count for the employee's role
- `remote_opportunity` — Prevalence of remote work in external job market

This hybrid approach provides a holistic view of why employees leave, incorporating both "push" (internal dissatisfaction) and "pull" (external opportunities) factors.

---

## 📊 Datasets

### Dataset Overview

| Dataset | Rows × Cols | Size | Source | Purpose |
|---------|-------------|------|--------|---------|
| **IBM HR Attrition** | 1,470 × 35 | ~1 MB | Kaggle | Internal HR baseline metrics & attrition labels |
| **LinkedIn Job Postings** | ~33,000 × 27 | ~10 MB | Kaggle | External market pull signals, salaries, demand |

### Downloading the Datasets

> **Note:** Raw data files are hosted externally on Kaggle. Download them manually using the links below and place them in the `data/raw/` directory.

#### 1. IBM HR Analytics Employee Attrition Dataset
- **Link:** [IBM HR Analytics Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **File to download:** `IBMHRAttritionDataset.csv` or `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Destination:** `filess/hr-attrition-analytics/hr-attrition-analytics/data/raw/`

#### 2. LinkedIn Job Postings Dataset
- **Link:** [LinkedIn Job Postings on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **File to download:** `job_postings.csv` or similar
- **Destination:** `filess/hr-attrition-analytics/hr-attrition-analytics/data/raw/`

### Data Storage Notes
- ⚠️ **Large files are NOT stored in this repository** to keep the repo lightweight
- Raw CSV files must be downloaded from Kaggle (see links above)
- Processed/merged datasets are generated during the pipeline execution
- All data handling follows Kaggle's Terms of Use

---

## 📁 Repository Structure

```
data-analytics-project/
│
├── README.md                          # This file
├── process_large_csv.py               # Utility script for CSV processing
├── run_analysis.py                    # Main execution script
├── requirements.txt                   # Python dependencies
├── small_linkedin_jobs.xlsx           # Sample LinkedIn data (reference)
│
└── filess/
    └── hr-attrition-analytics/
        └── hr-attrition-analytics/    # Main project directory
            │
            ├── data/
            │   ├── raw/               # 📥 Place downloaded Kaggle CSVs here
            │   │   ├── IBMHRAttritionDataset.csv    (to download)
            │   │   └── job_postings.csv             (to download)
            │   └── processed/         # Auto-generated merged & cleaned datasets
            │
            ├── docs/
            │   └── project_plan.md    # Detailed project methodology & timeline
            │
            ├── notebooks/             # Jupyter notebooks for interactive analysis
            │   ├── 01_data_loading.ipynb
            │   ├── 02_exploratory_analysis.ipynb
            │   ├── 03_feature_engineering.ipynb
            │   └── 04_model_training.ipynb
            │
            ├── reports/
            │   └── figures/           # Auto-generated plots & visualizations
            │
            ├── src/                   # Reusable Python modules
            │   └── *.py               # Feature engineering, modeling utilities
            │
            ├── README.md              # Project-specific documentation
            ├── requirements.txt       # Python package dependencies
            └── check_environment.py   # Environment verification script
```

### Directory Descriptions
- **`data/raw/`** — Download Kaggle datasets here (not in repo, download manually)
- **`data/processed/`** — Cleaned, merged, and feature-engineered datasets (auto-generated)
- **`notebooks/`** — Interactive Jupyter notebooks for EDA and analysis
- **`reports/figures/`** — Generated plots, confusion matrices, feature importance charts
- **`src/`** — Reusable Python modules for data pipelines and modeling
- **`docs/`** — Detailed project documentation and methodology

---

## 🔧 Methodology

### Phase 1: Data Collection & Loading
- Download raw datasets from Kaggle (IBM HR & LinkedIn)
- Load CSVs into pandas DataFrames
- Perform initial data validation and shape verification

### Phase 2: Exploratory Data Analysis (EDA)
- Analyze distributions of internal HR features
- Explore LinkedIn job market data patterns
- Identify missing values, outliers, and class imbalance
- Generate univariate and bivariate visualizations

### Phase 3: Feature Engineering & Data Integration
- **Internal Features:** Extract key HR metrics from IBM dataset
- **External Features:**
  - `salary_gap` — Calculate median salary for each role from LinkedIn, compare with internal
  - `external_demand_score` — Normalize job postings count per role
  - `remote_opportunity` — Compute proportion of remote jobs in each role
- **Data Merge:** Join datasets on job role mapping
- **Class Balancing:** Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle attrition imbalance

### Phase 4: Machine Learning & Modeling
Train and evaluate three models:

| Model | Purpose |
|-------|---------|
| **Logistic Regression** | Baseline interpretable linear model |
| **Random Forest** | Non-linear ensemble with feature importance |
| **XGBoost** | Gradient boosting for optimal performance |

Each model trained on two scenarios:
- **IBM-only dataset** — Internal features only
- **Hybrid dataset** — Internal + external features

### Phase 5: Evaluation & Visualization
- Compare performance metrics (Accuracy, F1-Score, AUC-ROC)
- Identify top predictive features
- Generate visualizations: confusion matrices, ROC curves, feature importance plots
- Document findings and validate hypotheses

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- ~2 GB free disk space (for raw datasets)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/bongshot/data-analytics-project.git
cd data-analytics-project
```

#### 2. Navigate to Project Directory
```bash
cd filess/hr-attrition-analytics/hr-attrition-analytics
```

#### 3. Create Virtual Environment
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Download Datasets
1. Visit [IBM HR Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Visit [LinkedIn Jobs Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
3. Download the CSV files
4. Place them in `data/raw/` directory:
   ```
   data/raw/IBMHRAttritionDataset.csv
   data/raw/job_postings.csv
   ```

#### 6. Verify Environment
```bash
python check_environment.py
```

Expected output:
```
✅ Python version OK
✅ Required packages installed
✅ Project structure verified
```

---

## ▶️ Running the Project

### Option 1: Run Interactive Notebooks (Recommended for Exploration)
```bash
jupyter notebook
```
Then open notebooks in this order:
1. `notebooks/01_data_loading.ipynb` — Load and inspect data
2. `notebooks/02_exploratory_analysis.ipynb` — EDA and visualization
3. `notebooks/03_feature_engineering.ipynb` — Feature creation & merging
4. `notebooks/04_model_training.ipynb` — Train and evaluate models

### Option 2: Run Full Pipeline (One Command)
```bash
python run_analysis.py
```
This executes all phases and generates reports in `reports/`

### Option 3: Run Individual Scripts
```bash
# Data loading
python src/data_loader.py

# Feature engineering
python src/feature_engineering.py

# Model training
python src/model_training.py
```

---

## 📈 Results

### Model Performance Comparison

| Model | Dataset | Accuracy | F1 Score | AUC-ROC |
|-------|---------|----------|----------|---------|
| **Logistic Regression** | IBM Only | 0.70 | 0.44 | 0.73 |
| **Logistic Regression** | IBM + LinkedIn | 0.71 | 0.45 | 0.74 |
| **Random Forest** | IBM Only | 0.86 | 0.35 | 0.76 |
| **Random Forest** | IBM + LinkedIn | 0.86 | 0.37 | 0.77 |
| **XGBoost** | IBM Only | 0.84 | 0.45 | 0.76 |
| **XGBoost** | IBM + LinkedIn | **0.86** | **0.48** | **0.78** |

### Key Findings

✅ **Hybrid Model Superiority**
- The XGBoost hybrid model achieved the best overall performance (AUC = 0.78)
- F1-score improved by 6.7% compared to IBM-only baseline

✅ **Novel Features Validate Pull-Push Theory**
- `salary_gap` ranked in the top 5 most important features
- Employees whose external market value significantly exceeds internal salary have **higher attrition risk**
- `external_demand_score` correlates positively with attrition

✅ **Balanced Insights**
- Both "push" factors (internal dissatisfaction) and "pull" factors (external opportunities) matter
- Salary competitiveness is crucial for retention
- Remote work availability affects employee decisions

---

## 📚 Citation

If you use this project, please cite:

```bibtex
@misc{hr_attrition_analytics_2026,
  author = {Student Name},
  title = {Beyond Internal HR Metrics: Predicting Employee Attrition Using a Hybrid Framework Integrating External Job Market Signals},
  year = {2026},
  publisher = {University of Dhaka},
  course = {K502 Business Analytics}
}
```

---

## 🙏 Acknowledgements

- **Professor:** Shah Mostafa Khaled, Ph.D. (Course K502 Business Analytics, University of Dhaka)
- **Data Sources:** 
  - IBM for HR Analytics dataset
  - Kaggle for hosting datasets and competition platform
- **Libraries:** pandas, scikit-learn, XGBoost, seaborn, matplotlib, imbalanced-learn

