# Customer Churn Prediction Model

A machine learning project that predicts customer churn for a telecom company using Logistic Regression, served via a Flask web application.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — contains 7,043 customer records with 21 features including demographics, account info, and services subscribed.

## Project Overview

The goal is to predict whether a customer will churn (leave the service) based on their profile and usage data. Early identification of at-risk customers allows businesses to take proactive retention measures.

## Features Used

- Customer demographics (gender, age, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, etc.)
- Charges (monthly and total)

## Tech Stack

- **Python 3**
- **pandas** — data loading and preprocessing
- **scikit-learn** — model training and evaluation
- **Flask** — web application framework
- **pickle** — model serialization

## Project Structure

```
customer-churn-model/
├── app.py                          # Flask web application
├── churn.py                        # Model training script
├── churn_model.pkl                 # Saved trained model
├── scaler.pkl                      # Saved StandardScaler
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── templates/
    └── index.html                  # Frontend UI
```

## Workflow

1. Load and clean the dataset (handle missing values in `TotalCharges`)
2. Drop irrelevant columns (`customerID`)
3. Encode categorical variables using one-hot encoding (`pd.get_dummies`)
4. Scale features using `StandardScaler`
5. Train a Logistic Regression model
6. Evaluate using classification report, confusion matrix, and accuracy score
7. Save the trained model and scaler using `pickle`
8. Serve predictions via a Flask web app

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Precision (Churn) | 0.62 |
| Recall (Churn) | 0.52 |
| F1-Score (Churn) | 0.56 |

## How to Run

1. Clone the repo
```bash
git clone https://github.com/Clawoid/Customer-Churn-Model.git
cd Customer-Churn-Model
```

2. Install dependencies
```bash
pip install pandas scikit-learn flask
```

3. Train the model (generates `churn_model.pkl` and `scaler.pkl`)
```bash
python churn.py
```

4. Run the Flask app
```bash
python app.py
```

5. Open your browser and go to `http://127.0.0.1:5000`
