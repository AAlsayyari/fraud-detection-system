# Fraud Detection System

A machine learning project to detect financial fraud in real time using the PaySim dataset. Built as a way to get hands on experience with the kind of problems FinTech solves, specifically fraud detection, transaction monitoring, and risk scoring.

## Why I Built This

I've been really interested in how AI is used in financial crime prevention. I wanted to build something from scratch to understand the full pipeline from raw transaction data all the way to a working dashboard that can flag suspicious activity in real time.

The dataset I used simulates mobile money transactions and contains over 6 million records with labeled fraud cases. Only about 0.3% of transactions are actually fraudulent, which makes it a pretty realistic scenario and a good challenge for machine learning.

## What I Did

### 1. Exploratory Data Analysis (`notebooks/EDA.ipynb`)
- Analyzed transaction types and discovered that fraud only occurs in TRANSFER and CASH_OUT transactions
- This alone cuts down the data we need to process by about 60%
- Found that 97.8% of fraudulent transactions empty the sender's entire balance
- Engineered two features that turned out to be really important:
  - `is_balance_emptied` whether the sender transferred their full balance
  - `dest_error` the mismatch between expected and actual receiver balance (catches cases where money disappears)

### 2. Model Training (`notebooks/machineLearning.ipynb`)
Tested three models:

| Model | F1-Score (Fraud) | AUPRC |
|-------|-----------------|-------|
| Logistic Regression | 0.41 | 0.42 |
| Random Forest | 0.99 | 0.99 |
| XGBoost | 0.99 | 0.99 |

The key thing I learned was about class imbalance. My first attempt with XGBoost gave much worse results (F1 = 0.92) because I didn't account for the fact that fraud cases are 336x rarer than legitimate ones. After setting scale_pos_weight to handle this, XGBoost jumped to 0.99 AUPRC.

I also ran cross validation to make sure the models weren't just memorizing the training data, both Random Forest and XGBoost held up with consistent scores across folds.

The final model is XGBoost, and I also save the LabelEncoder, feature names, and optimal threshold alongside it so the dashboard uses the exact same preprocessing.

### 3. Streamlit Dashboard (`notebooks/dashboard.py`)
A live dashboard where you can input transaction details and get an instant fraud prediction. It includes:
- Preloaded examples of fraudulent and legitimate transactions (both TRANSFER and CASH_OUT)
- Risk score with the model's confidence level
- A breakdown of what the model analyzed and why it flagged something

## Project Structure

```
fraud-detection-system/
├── data/
│   └── PS_20174392719_1491204439457_log.csv   # PaySim dataset (~470MB)
├── notebooks/
│   ├── EDA.ipynb                               # Exploratory analysis
│   ├── machineLearning.ipynb                   # Model training & evaluation
│   └── dashboard.py                            # Streamlit dashboard
├── src/
│   ├── fraud_model.pkl                         # Trained XGBoost model
│   ├── label_encoder.pkl                       # Saved LabelEncoder
│   ├── feature_names.pkl                       # Feature ordering
│   └── optimal_threshold.pkl                   # Detection threshold
└── README.md
```

## How to Run

**Requirements:** Python 3.10+, plus these packages:
```
pip install pandas numpy scikit-learn xgboost joblib streamlit seaborn matplotlib
```

**Run the dashboard:**
```
python -m streamlit run notebooks/dashboard.py
```
Then open http://localhost:8501 in your browser.

**Train the model yourself:**
1. Download the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in `data/`
2. Run all cells in `notebooks/machineLearning.ipynb`

## What I Learned

- Class imbalance is a huge deal in fraud detection. Accuracy can look great (99.7%) even if the model catches zero fraud — you need metrics like AUPRC and F1 that actually measure how well you find the rare cases.
- Feature engineering matters more than model complexity. The two features I built from the EDA (`is_balance_emptied` and `dest_error`) are what really drove the model's performance.
- Saving preprocessing artifacts (encoders, thresholds) alongside the model is important. My first version had bugs because the dashboard was hardcoding values that didn't match what the model was trained on.

## Dataset

[PaySim: Synthetic Financial Dataset for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) simulates mobile money transactions based on real transaction patterns from an African mobile money service.
