# Customer Churn Prediction

Entry for a hackathon-style churn prediction contest: predict merchant customer
churn from account/usage features. Seven model versions (`submit1.py` ..
`submit7.py`), each adding features or swapping the classifier (logistic
regression → KNN → random forest → XGBoost), plus rule-based post-adjustment
of predictions using each merchant's other records.

## Data

`train.csv.gz` (full set) and its `train70.csv` / `test30.csv` split are
account-window snapshots with a binary `churned` label. Merchant identity has
been anonymized: `contract_id` and `merchant_name` are sequential placeholders
(`CNT000123`, `Merchant 000123`) with no link back to the original accounts.
Some scripts derive text features from `merchant_name` (word count, "ltd"/
"tech"/"capital" substring flags, etc.) — these are inert on the anonymized
data since the placeholders carry none of that signal.

## Running

```
make install
make run N=7   # trains, evaluates, prints F1 + confusion matrix
```

Each script is self-contained: it trains on `train.csv.gz`, saves a
`model_submitN.joblib`, then evaluates against `test30.csv`. Model files
aren't checked in — retrain to regenerate them.

## Disclaimer

None of the data in this repo reflects real accounts, contracts, or usage.
Every field — identities, dates, and all numeric values — has been
diffused and randomized; nothing here can be traced back to any real
company or individual. It's shaped to resemble a plausible churn-prediction
scenario for the purpose of the contest, not to represent actual data.
