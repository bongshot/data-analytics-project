# PROJECT TITLE
Beyond Internal HR Metrics: Predicting Employee Attrition Using a Hybrid HR-Job Market Analytics Framework

## TEAM MEMBERS
Student | Course: K502 Business Analytics

## ABSTRACT
Employee attrition is a massive cost center for businesses, leading to loss of knowledge, productivity drops, and expensive rehiring processes. While most predictive modeling focuses exclusively on internal HR features (such as job satisfaction, distance from home, and salary), this project proposes a novel approach that integrates external job market pull factors. By joining IBM HR attrition data with scraped LinkedIn job postings, we evaluate the impact of external salary gaps, remote work availability, and industry demand on an employee's likelihood to leave. Our findings demonstrate that combining internal and external metrics significantly improves predictive performance.

## PROBLEM STATEMENT
Organizations struggle to proactively identify employees at risk of leaving. Currently, most models rely solely on internal HR data, missing a critical driver of attrition: the external job market. When an employee is underpaid relative to the external market or when demand for their role surges externally, their attrition risk increases regardless of internal satisfaction. This project addresses this gap by creating a hybrid model that accounts for both internal push factors and external pull factors.

## RESEARCH QUESTIONS
1. Which internal HR factors most strongly predict employee attrition?
2. Do external job market signals (salary benchmarks, remote availability, job demand) independently predict attrition when controlling for internal factors?
3. Does a hybrid model (internal + external features) significantly outperform an internal-only model in attrition prediction?

## NOVELTY STATEMENT
While many studies have applied machine learning to the IBM HR dataset to predict attrition, they universally limit their scope to internal variables. This project is original because it bridges internal HR data with external labor market data. We introduce three novel features: `salary_gap` (external median vs. internal salary), `external_demand_score` (normalized applicant counts for the role), and `remote_opportunity` (prevalence of remote work in external postings). This hybrid approach provides a more holistic view of the employee experience.

## DATASETS
| Dataset | Description | Purpose |
|---------|-------------|---------|
| **IBM HR Dataset** | 1,470 records, 35 features | Used to establish the internal baseline for attrition prediction. |
| **LinkedIn Job Postings** | ~33,000 records, 27 features | Used to compute external market features (salary, demand, remote). |

Integrating these datasets enables the computation of market-relative features, forming the core novelty of our approach.

## METHODOLOGY
- **Phase 1:** Data collection and loading from raw sources.
- **Phase 2:** Exploratory Data Analysis (EDA) on both datasets individually.
- **Phase 3:** Feature engineering and merging. The three novel features are computed and the datasets are joined based on a job role mapping. SMOTE is applied to handle class imbalance.
- **Phase 4:** Modeling using Logistic Regression, Random Forest, and XGBoost to compare the IBM-only baseline against the hybrid model.
- **Phase 5:** Evaluation, visualization of results, and drafting the final paper.

## EXPECTED RESULTS
We hypothesize that `salary_gap` and `external_demand_score` will rank among the most important predictive features in the XGBoost model. Additionally, we expect the hybrid model to improve the F1-score by at least 5 percentage points over the baseline model, validating the inclusion of external market data.

## TOOLS & TECHNOLOGIES
Python, pandas, scikit-learn, XGBoost, imbalanced-learn, seaborn, matplotlib, VS Code, Git/GitHub.

## TIMELINE
| Phase | Task | Status |
|-------|------|--------|
| 1 | Data Setup & Loading | Complete |
| 2 | Exploratory Data Analysis | Complete |
| 3 | Feature Engineering & Merge | Complete |
| 4 | Model Training & Evaluation | Complete |
| 5 | Visualization & Reporting | Complete |

## REFERENCES
1. Zhao, Y., et al. (2020). "Predicting Employee Turnover with Machine Learning." *IEEE Access*.
2. Hom, P. W., et al. (2019). "Employee Turnover: Push and Pull Factors." *Journal of Applied Psychology*.
3. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*.
4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
5. IBM Corporation (2018). "IBM HR Analytics Employee Attrition & Performance." *Kaggle Dataset*.
