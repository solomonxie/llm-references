import joblib
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

import xgboost

FEATURES = [
    # Basic
    # 'merchant_name',
    # 'contract_id',

    # Account
    'contract_age',
    'user_count',
    'mfa_enabled',
    'AvgUniqueUsers',

    # Support
    'supportTicketCount',
    'minTicketSentiment',
    'avgTicketSentiment',
    'maxTicketOpenTime',

    # Sales
    'new_business_revenue_in_window',
    'metric_bundle_count',
    'renewals_in_lookahead',
    'logins_in_window',
    'renewal_revenue_in_window',
    'upsell_revenue_in_window',

    # Web
    'web_access_enabled',
    'SessionCount',
    'PageviewCount',
    'CompareGroupCount',
    'PageviewSlope',
    'SessionSlope',
    'CompareGroupSlope',

    # Web
    'api_access_enabled',
    'api_calls_per_month',

    # CSV
    'csv_export_enabled',
    'ScheduledReportSlope',
    'SavedReportSlope',
    'SavedReportCount',
]

DATE_COLS = [
    'window_start',
    'window_end',
    'lookahead_start',
    'lookahead_end',
    'contract_start_date',
]

DTYPES = {
    'window_start': 'datetime64',
    'window_end': 'datetime64',
    'lookahead_start': 'datetime64',
    'lookahead_end': 'datetime64',
    'contract_start_date': 'datetime64',
}

PARAMS = {
    # NUMERIC
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'max_depth': 50,
    # 'min_child_weight': 5,
    # 'max_delta_step': 1,
    # 'colsample_bytree': 0.1,

    # CHOICES
    'booster': 'gbtree',
    'objective': 'binary:hinge',
    # 'normalize_type': 'forest',
    # 'sample_type': 'weighted',

    # FIXED
    'nthread': 1,
    'gamma': 0,  # Larger gamma => more conservative
    'subsample': 0.5,  # Ratio of the training instances
    'lambda': 1,  # L2 regularization
    'alpha': 0,  # L1 regularization
    'sampling_method': 'uniform',
    # 'scale_pos_weight': scale_pos_weight,
    # 'top_k': 10,  # The number of top features to select
    # 'seed': 99,
}


class MyModel:

    def generate_x(self, input_data):
        df = input_data.astype(DTYPES)
        df['n_subscribed'] = df['contract_start_date'].apply(
            lambda x: int((x - datetime(1900, 1, 1)).total_seconds())
        )
        df['n_agg'] = df['window_start'].apply(
            lambda x: int((x - datetime(1900, 1, 1)).total_seconds())
        )
        df['merchant_name_len'] = df['merchant_name'].apply(lambda x: len(x))
        # cidmap = {contractid: i for i, contractid in enumerate(df['contract_id'])}
        # df['contractid'] = df['contract_id'].apply(lambda x: cidmap.get(x) or 0)
        # df['merchant_name_parts'] = df['merchant_name'].apply(lambda x: len(x.replace(',', ' ').split()))
        features = FEATURES + ['n_subscribed', 'n_agg', 'merchant_name_len']
        X = df[features]
        imputed = SimpleImputer(strategy="mean")  # Impute missing value as mean of column
        X = imputed.fit_transform(X)
        return X

    def fit(self, input_data, y):
        print('[ TRAINING ]...')
        X = self.generate_x(input_data)
        self.model = xgboost.XGBClassifier(**PARAMS)
        self.model.fit(X, y)
        return None

    def predict(self, input_data):
        X = self.generate_x(input_data)
        predicted_y = self.model.predict(X)
        return predicted_y


def train():
    df = pd.read_csv('train.csv.gz')
    # df = pd.read_csv('train70.csv')
    # balanced = pd.concat([df, *[df[df.churned == 1] for _ in range(10)]])
    # df = balanced
    target = df['churned']
    model = MyModel()
    model.fit(df, target)
    joblib.dump(model, 'model_submit4.joblib')
    print('[ OK ] Saved model to: model_submit4.joblib')


def test():
    split_csv()
    # Load holdout test data
    holdout_data = pd.read_csv('test30.csv')
    # holdout_data = pd.read_csv('train.csv.gz')
    holdout_X = holdout_data
    y = holdout_data['churned']

    # Load model
    model = joblib.load('model_submit4.joblib')

    # Predict with trained model
    predicted_y = model.predict(holdout_X)

    # Result
    f1score = f1_score(y, predicted_y)
    print("The model's F1-score on holdout set is", f1score)
    (_0_0, _0_1), (_1_0, _1_1) = confusion_matrix(y, predicted_y)
    print(f'The confusion matrix is: 1-1 ({_1_1}), 0-0 ({_0_0}), 1-0 ({_1_0}), 0-1 ({_0_1})')

    joblib.dump(model, f'/tmp/model_submit4_{f1score:.4}.joblib')


def split_csv(k=0.5):
    df = pd.read_csv('train.csv.gz')
    train, test = train_test_split(df, test_size=k)
    train.to_csv('train70.csv', index=False, header=True)
    test.to_csv('test30.csv', index=False, header=True)


if __name__ == '__main__':
    train()
    test()
