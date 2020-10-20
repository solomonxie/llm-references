import joblib
import pandas as pd
from datetime import datetime
from collections import Counter
from dateutil.relativedelta import relativedelta

# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

import xgboost

FEATURES = [
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
    'n_estimators': 2500,
    'max_depth': 40,
    'gamma': 0,  # (min_split_loss) Larger gamma => more conservative
    'lambda': 1,  # L2 regularization
    'alpha': 1,  # L1 regularization
    'subsample': 0.8,  # Ratio of the training instances

    # CHOICES
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'grow_policy': 'lossguide',  # 'depthwise', 'lossguide'
    'sampling_method': 'uniform',
    'nthread': 8,
    'verbosity': 1,
}


def to_date(sdate):
    return datetime.strptime(sdate, '%Y-%m-%d')


def generate_x(df):
    # Days
    df['subscribed_days'] = df['contract_start_date'].apply(
        lambda x: int((to_date(x) - datetime(1900, 1, 1)).total_seconds())
    )
    df['window_days'] = df['window_start'].apply(
        lambda x: int((to_date(x) - datetime(1900, 1, 1)).total_seconds())
    )
    df['contract_age_months'] = df['contract_start_date'].apply(
        lambda x: int(relativedelta(to_date(x), datetime(1900, 1, 1)).months)
    )
    df['window_months'] = df['window_start'].apply(
        lambda x: int(relativedelta(to_date(x), datetime(1900, 1, 1)).months)
    )

    # Names
    df['merchant_name_lower'] = df['merchant_name'].apply(lambda x: x.lower())
    df['merchant_name_len'] = df['merchant_name'].apply(lambda x: len(x))
    df['merchant_name_parts'] = df['merchant_name'].apply(lambda x: len(x.replace(',', ' ').replace('.', ' ').split()))
    df['n_merchant_name_dot'] = df['merchant_name'].apply(lambda x: Counter(x).get('.') or 0)
    df['is_ltd'] = df['merchant_name_lower'].apply(lambda x: any(['ltd' in x, 'limited' in x, 'llc' in x]))
    df['is_tech'] = df['merchant_name_lower'].apply(
        lambda x: any(['tech' in x, 'soft' in x, 'dev' in x, 'digit' in x, 'solution' in x, '.com' in x])
    )
    df['is_game'] = df['merchant_name_lower'].apply(lambda x: any(['game' in x]))
    df['is_capital'] = df['merchant_name_lower'].apply(
        lambda x: any(['capital' in x, 'investiment' in x, 'bank' in x, 'asset' in x, 'insurance' in x])
    )
    df['is_inc'] = df['merchant_name_lower'].apply(lambda x: any(['inc.' in x]))
    df['is_asia'] = df['merchant_name_lower'].apply(lambda x: any(['asia' in x]))
    df['is_hk'] = df['merchant_name_lower'].apply(lambda x: any(['hong kong' in x, 'hk' in x]))
    df['is_bj'] = df['merchant_name_lower'].apply(lambda x: any(['beijing' in x, 'bj' in x]))
    df['is_cn'] = df['merchant_name_lower'].apply(lambda x: any(['china' in x]))
    df['is_us'] = df['merchant_name_lower'].apply(lambda x: any(['u.s' in x]))
    df['is_jp'] = df['merchant_name_lower'].apply(lambda x: any(['jp' in x, 'japan' in x]))
    df['is_digital'] = df['merchant_name_lower'].apply(lambda x: any(['digital' in x]))
    df['is_media'] = df['merchant_name_lower'].apply(lambda x: any(['media' in x, 'entertian' in x, 'music' in x]))
    df['is_management'] = df['merchant_name_lower'].apply(lambda x: any(['management' in x]))
    df['is_public'] = df['merchant_name_lower'].apply(lambda x: any(['public' in x]))
    df['is_private'] = df['merchant_name_lower'].apply(lambda x: any(['private' in x]))
    df['is_corp'] = df['merchant_name_lower'].apply(lambda x: any(['corp' in x]))
    # cidmap = {contractid: i for i, contractid in enumerate(df['contract_id'])}
    # df['contractid'] = df['contract_id'].apply(lambda x: cidmap.get(x) or 0)

    # Bins
    df['user_bins'] = pd.cut(df['user_count'], bins=10, labels=False)
    df['account_age_bins'] = pd.cut(df['contract_age'], bins=10, labels=False)
    df['n_bundle_bins'] = pd.cut(df['metric_bundle_count'], bins=5, labels=False)
    df['n_renew_bins'] = pd.cut(df['renewals_in_lookahead'], bins=10, labels=False)
    df['page_view_bins'] = pd.cut(df['PageviewCount'], bins=10, labels=False)
    df['compare_group_bins'] = pd.cut(df['CompareGroupCount'], bins=10, labels=False)

    df.to_csv('/tmp/features.csv', index=False, header=True)
    features = [
        # Basic
        # 'merchant_name',
        # 'contract_id',

        # Account
        'contract_age',  # -0.13
        'user_count',  # 0.17
        'mfa_enabled',  # 0.033
        # 'AvgUniqueUsers',  # -0.064 (correlated many)

        # Support
        'supportTicketCount',  # -0.078
        'minTicketSentiment',  # 0.07
        'avgTicketSentiment',  # -0.043
        'maxTicketOpenTime',  # -0.043

        # Sales
        'new_business_revenue_in_window',  # -0.012
        'metric_bundle_count',  # -0.21
        'renewals_in_lookahead',  # 0.44
        'logins_in_window',  # -0.19
        'renewal_revenue_in_window',  # -0.083
        'upsell_revenue_in_window',  # -0.071

        # Web
        'web_access_enabled',  # -0.051
        # 'SessionCount',  # -0.085
        'PageviewCount',  # -0.08  (keep)
        'CompareGroupCount',  # -0.076  (keep)
        # 'PageviewSlope',  # -0.0016
        'SessionSlope',  # -0.035
        'CompareGroupSlope',  # -0.071  (keep)

        # Web
        'api_access_enabled',  # -0.025
        'api_calls_per_month',  # -0.053

        # CSV
        'csv_export_enabled',  # -0.11
        'ScheduledReportSlope',  # -0.053
        'SavedReportSlope',  # -0.072
        'SavedReportCount',  # -0.076  (keep)

        # ==============FEATURE TRANSFORMATION============>
        # 'merchant_name_len',  # 0.0076
        'merchant_name_parts',  # 0.011  (corr merchant_name_len)
        'n_merchant_name_dot',  # -0.022
        # 'window_days',  # 0.0023 (bad)
        'subscribed_days',  # 0.13 (corr contract_age)
        # 'window_months',
        # 'contract_age_months',
        'is_corp',  # -0.015
        'is_tech',  # -0.013
        'is_game',  # -0.013
        'is_capital',  # -0.012
        'is_ltd',  # 0.029
        'is_inc',  # -0.028
        'is_management',  # -0.016
        # 'is_private',  # -0.018
        # 'is_public',  # -0.0026 (bad)
        # 'is_digital',  # -0.019
        # 'is_media',  # 0.018

        # 'is_asia',  # 0.022
        # 'is_hk',  # 0.02
        # 'is_bj',  # 0.041
        # 'is_cn',  # 0.02
        # 'is_us',  # 0.037
        # 'is_jp',  # 0.021

        # 'user_bins',  # 0.16 (bad)
        # 'account_age_bins',  # -0.14 (bad)
        # 'n_bundle_bins',  # -0.19 (bad)
        # 'n_renew_bins',  # 0.33 (bad)
        # 'page_view_bins',  # -0.037 (bad)
        # 'compare_group_bins',  # -0.025 (bad)
    ]

    X = df[features]
    imputed = SimpleImputer(strategy="mean")  # Impute missing value as mean of column
    X = imputed.fit_transform(X)

    return X


def adjust_y(train_data, test_X, predicted_y):
    train = train_data
    train['start'] = train['lookahead_start'].apply(lambda x: to_date(x))
    train['end'] = train['lookahead_end'].apply(lambda x: to_date(x))
    test = test_X.copy()
    test['start'] = test['lookahead_start'].apply(lambda x: to_date(x))
    test['end'] = test['lookahead_end'].apply(lambda x: to_date(x))
    test['churned'] = None
    test['predicted_y'] = predicted_y
    adjusted_y = predicted_y
    test['from'] = 'test'
    test['ind'] = range(len(train), len(train) + len(test))
    full = pd.concat([train, test])
    assert len(full) == len(train) + len(test)

    # 1. The latest STAY record: all the records' ch_end before its ch_end should be STAY
    print('Adjusting Rule-1...')
    for i, row in test.iterrows():
        match = train[(train['contract_id'] == row['contract_id']) & (train['churned'] == 0)]
        if match.empty:
            continue
        latest_stay = match.sort_values('end').iloc[-1]['end']
        if row['end'] < latest_stay and adjusted_y[i] == 1:
            print('\t adjusted: 1->0')
            adjusted_y[i] = 0

    # 2. The earliest CHURN record: all the records' churn_start after its churn_start should be CHURN
    print('Adjusting Rule-2...')
    for i, row in test.iterrows():
        match = train[(train['contract_id'] == row['contract_id']) & (train['churned'] == 1)]
        if match.empty:
            continue
        earliest_churn = match.sort_values('start').iloc[-1]['start']
        if row['start'] > earliest_churn and adjusted_y[i] == 0:
            print('\t adjusted: 0->1')
            adjusted_y[i] = 1

    # 3. If there're 4 more rec after the given date, the given row must be 0
    print('Adjusting Rule-3...')
    for i, row in test.iterrows():
        match = full[(full['contract_id'] == row['contract_id']) & (full['start'] > row['start'])]
        if len(match) >= 4 and adjusted_y[i] == 1:
            print('\t adjusted: 1->0')
            adjusted_y[i] = 0

    return adjusted_y


class MyModel:

    def fit(self, train_X, train_y):
        print('[ TRAINING ]...')
        self.train_X = train_X
        self.train_y = train_y
        self.train_data = train_X.copy()
        self.train_data['churned'] = train_y
        self.train_data['predicted_y'] = train_y
        self.train_data['from'] = 'train'
        self.train_data['ind'] = range(0, len(self.train_data))
        X = generate_x(self.train_data)
        self.model = xgboost.XGBClassifier(**PARAMS)
        self.model.fit(X, self.train_y)
        return None

    def predict(self, test_data):
        X = generate_x(test_data)
        predicted_y = self.model.predict(X)
        adjusted_y = adjust_y(self.train_data, test_data, predicted_y)
        return adjusted_y


def train():
    df = pd.read_csv('train.csv.gz')
    # df = pd.read_csv('train70.csv')
    target = df['churned']
    model = MyModel()
    model.fit(df, target)
    joblib.dump(model, 'model_submit5.joblib')
    print('[ OK ] Saved model to: model_submit5.joblib')


def test():
    test_data = pd.read_csv('test30.csv')

    holdout_X = test_data
    y = test_data['churned']
    # Load model
    model = joblib.load('model_submit5.joblib')
    # Predict with trained model
    predicted_y = model.predict(holdout_X)
    # Result
    f1score = f1_score(y, predicted_y)
    print("The model's F1-score on holdout set is", f1score)
    (_0_0, _0_1), (_1_0, _1_1) = confusion_matrix(y, predicted_y)
    print(f'The confusion matrix is: 1-1 ({_1_1}), 0-0 ({_0_0}), 1-0 ({_1_0}), 0-1 ({_0_1})')
    joblib.dump(model, f'/tmp/model_submit5_{f1score:.4}.joblib')

    test_data['predict'] = predicted_y
    test_data.to_csv('/tmp/predict.csv', index=False, header=True)


def split_csv(k=0.3):
    df = pd.read_csv('train.csv.gz')
    train, test = train_test_split(df, test_size=k)
    train.to_csv('train70.csv', index=False, header=True)
    test.to_csv('test30.csv', index=False, header=True)


def main():
    # split_csv()
    train()
    test()


if __name__ == '__main__':
    main()

    # Correlation matrix
    # font = {'size': 20}
    # plt.rc('font', **font)
    # plt.figure(figsize=(50, 25))
    # plt.title('Corralation Matrix', size=30)
    # sns.heatmap(pd.read_csv('train.csv.gz').corr(), annot=True, linewidths=0.1)
    # # plt.show()
    # plt.savefig('/tmp/correlation-matrix.png')
    # print('Saved correlation-matrix to file: /tmp/correlation-matrix.png')
