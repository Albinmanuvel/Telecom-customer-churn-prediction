# Telecom customer churn prediction

Logistic regression to predict which telecom customers will churn. An early learning project, kept as a record.

## Data
`Telecom_Data.csv` - 3,333 customers, 20 features (plan types, day/evening/night/international usage, customer service calls) and a `churn` flag.

## What the script does (`churn-analysis.py`)
1. Feature selection with Welch ANOVA and Pearson correlation against churn.
2. 70/30 train/test split (`random_state=1`).
3. Logistic regression, evaluated with accuracy, confusion matrix and classification report.

## Results (test set, 1,000 rows)
- Accuracy: 0.851
- Churn class: precision 0.46, recall 0.18, F1 0.26

Accuracy looks fine but is misleading: only about 14% of customers churn, so a model that predicts "no churn" for everyone would already score about 85%. This model catches only 18% of real churners. Next steps would be class weights or resampling, a tree-based model, and judging by recall / PR-AUC instead of accuracy.

## Run it
```
pip install -r requirements.txt
python churn-analysis.py
```
