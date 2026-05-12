# Beyond Internal HR: Predicting Employee Attrition by Merging HR Analytics with External Job Market Signals
![Python Badge](https://img.shields.io/badge/Python-3.10%2B-blue)
![License Badge](https://img.shields.io/badge/License-MIT-green)
![Status Badge](https://img.shields.io/badge/Status-Active-success)

A novel data mining approach to HR Analytics integrating internal HR records with external LinkedIn job market data.

## Abstract
Employee attrition is a critical challenge for modern organizations, costing billions annually in lost productivity and hiring expenses. Traditional predictive models rely exclusively on internal HR metrics. This project proposes a novel approach by merging internal IBM HR data with external job market signals from LinkedIn. By engineering features such as salary gap, external job demand, and remote opportunity, we successfully demonstrate that external pull factors significantly improve attrition prediction. 

## Datasets
| Dataset | Rows × Cols | Source | Purpose |
|---------|-------------|--------|---------|
| IBM HR Attrition | 1,470 × 35 | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) | Internal HR baseline metrics |
| LinkedIn Jobs | ~33,000 × 27 | [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | External market pull signals |

## Repository Structure
```
hr-attrition-analytics/
├── data/
│   ├── raw/             # Original CSVs (IBM and LinkedIn)
│   └── processed/       # Merged and cleaned dataset
├── docs/                # Project plans and documentation
├── notebooks/           # Jupyter notebooks for interactive analysis
├── reports/
│   └── figures/         # Auto-generated plots and visualizations
├── src/                 # Reusable Python source code modules
├── README.md            # Project overview and instructions
├── requirements.txt     # Python dependencies
└── check_environment.py # Verification script
```

## Setup & Installation
1. Clone the repository: `git clone <repository-url>`
2. Navigate to project: `cd hr-attrition-analytics`
3. Create virtual environment: `python -m venv venv`
4. Activate: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Verify setup: `python check_environment.py`

## Experiments & Results
| Model | Dataset | Accuracy | F1 Score | AUC |
|-------|---------|----------|----------|-----|
| Logistic Regression | IBM Only | 0.70 | 0.44 | 0.73 |
| Logistic Regression | IBM + LinkedIn | 0.71 | 0.45 | 0.74 |
| Random Forest | IBM Only | 0.86 | 0.35 | 0.76 |
| Random Forest | IBM + LinkedIn | 0.86 | 0.37 | 0.77 |
| XGBoost | IBM Only | 0.84 | 0.45 | 0.76 |
| XGBoost | IBM + LinkedIn | 0.86 | 0.48 | 0.78 |

## Key Findings
- The XGBoost hybrid model achieved the best performance with a significant improvement over the baseline.
- Novel feature `salary_gap` ranked in the top 5 most important features.
- Employees whose external market value significantly exceeds their current salary have a higher attrition risk.
- Higher `external_demand_score` (many jobs available) correlates positively with attrition.
- The hybrid approach confirms the "Pull-Push" theory of turnover.

## Citation
```bibtex
@misc{hr_attrition_analytics_2026,
  author = {Student Name},
  title = {Beyond Internal HR Metrics: Predicting Employee Attrition Using a Hybrid Framework},
  year = {2026},
  publisher = {University of Dhaka}
}
```

## Acknowledgements
Thanks to Professor Shah Mostafa Khaled, Ph.D. for guidance in K502 Business Analytics at the University of Dhaka. Datasets provided by IBM and Kaggle.
