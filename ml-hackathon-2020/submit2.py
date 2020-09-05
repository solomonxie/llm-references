import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix

FEATURES = [
    # 0.5
    'new_business_revenue_in_window',
    'metric_bundle_count',
    'renewals_in_lookahead',
    'logins_in_window',
    'renewal_revenue_in_window',
    'upsell_revenue_in_window',
    # 0.7
    'contract_age',
    # 0.9
    'user_count',
]


class MyModel:
    def __init__(self):
        self.features = FEATURES

    def generate_x(self, input_data):
        X = input_data[self.features]  # .sort_values(self.features[1])
        return X

    def fit(self, input_data, y):
        X = self.generate_x(input_data)
        highest_score = 0
        best_model = None
        print('Training....')
        for k in range(1, 20):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)
            f1score = self.test_model(model)
            if f1score > highest_score:
                highest_score = f1score
                best_model = model
                print(f'\t Trained KNN model (k={k}) with F-1 Score at: {f1score}')
        input_data.to_csv(f'/tmp/train_{highest_score:.2}.csv', index=False, header=True)
        print('Also saved training data to: /tmp/train_xx.csv')
        self.model = best_model
        return highest_score

    def test_model(self, model):
        split_csv()
        dtest = pd.read_csv('test30.csv')
        X = self.generate_x(dtest)
        y = dtest['churned']
        predicted_y = model.predict(X)
        f1score = f1_score(y, predicted_y)
        return f1score

    def predict(self, input_data):
        X = self.generate_x(input_data)
        predicted_y = self.model.predict(X)
        return predicted_y


def train():
    df = pd.read_csv('train.csv.gz')
    # df = pd.read_csv('train70.csv')
    input_data = df
    target = df['churned']
    model = MyModel()
    f1score = model.fit(input_data, target)
    joblib.dump(model, 'model_submit2.joblib')
    joblib.dump(model, f'/tmp/model_submit2_{f1score:.4}.joblib')
    print('[ OK ] Saved model to: model_submit2.joblib')


def test():
    split_csv()
    # Load holdout test data
    holdout_data = pd.read_csv('test30.csv')
    # holdout_data = pd.read_csv('train.csv.gz')
    holdout_X = holdout_data
    y = holdout_data['churned']

    # Load model
    model = joblib.load('model_submit2.joblib')

    # Predict with trained model
    predicted_y = model.predict(holdout_X)

    # Result
    f1score = f1_score(y, predicted_y)
    print("The model's F1-score on holdout set is", f1score)
    (_0_0, _0_1), (_1_0, _1_1) = confusion_matrix(y, predicted_y)
    print(f'The confusion matrix is: 1-1 ({_1_1}), 0-0 ({_0_0}), 1-0 ({_1_0}), 0-1 ({_0_1})')
    joblib.dump(model, f'/tmp/model_submit2_{f1score:.4}.joblib')


def split_csv(k=0.9):
    df = pd.read_csv('train.csv.gz')
    train, test = train_test_split(df, test_size=k)
    train.to_csv('train70.csv', index=False, header=True)
    test.to_csv('test30.csv', index=False, header=True)


if __name__ == '__main__':
    train()
    test()
