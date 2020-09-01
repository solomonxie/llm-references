import joblib
import random
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


class MyModel:
    def generate_x(self, input_data):
        features = [
            'renewals_in_lookahead',  # 0.45
            'contract_age',  # -0.14
            'user_count',  # 0.16
            'metric_bundle_count',  # -0.22
            'logins_in_window',  # -0.19
        ]
        X = input_data[features]
        return X

    def fit(self, input_data, y):
        X = self.generate_x(input_data)
        self.model = LogisticRegression(class_weight="balanced", solver='liblinear')
        self.model.fit(X, y)

    def predict(self, input_data):
        X = self.generate_x(input_data)
        predicted_y = self.model.predict(X)
        return predicted_y

    def filter_uncertain(self, input_data):
        # Eliminate certain results
        ymap = {}
        certain_set = set()
        for i, row in input_data.iterrows():
            conditions = [
                # Non-zero sales
                row['renewal_revenue_in_window'] > 0,
                row['upsell_revenue_in_window'] > 0,
                # Non-zero API calls
                row['api_calls_per_month'] > 0
            ]
            if any(conditions):
                certain_set.add(i)
                ymap[i] = 1
            else:
                ymap[i] = None

        # certain_list = list(certain_set)
        uncertan_list = list(set(range(len(input_data))) - certain_set)
        return ymap, uncertan_list

    def fit_after_elimination(self, input_data, y):
        _, uncertain_list = self.filter_uncertain(input_data)
        uncertain_X = input_data.iloc[uncertain_list]
        uncertain_y = [y[i] for i in uncertain_list]
        X = self.generate_x(uncertain_X)
        self.model = LogisticRegression(class_weight="balanced", solver='liblinear')
        self.model.fit(X, uncertain_y)

    def predict_after_elimination(self, input_data):
        ymap, uncertain_list = self.filter_uncertain(input_data)
        uncertain_X = input_data.iloc[uncertain_list]
        X = self.generate_x(uncertain_X)
        predictions = self.model.predict(X)
        for i, index in enumerate(uncertain_list):
            ymap[index] = predictions[i]

        predicted_y = [ymap[k] for k in range(len(input_data))]
        return predicted_y


def train():
    df = pd.read_csv('train70.csv')
    input_data = df.drop('churned', 1)
    target = df['churned']
    model = MyModel()
    # model.fit(input_data, target)
    model.fit_after_elimination(input_data, target)
    joblib.dump(model, 'model_submit1.joblib')
    print('[ OK ] Saved model to: model_submit1.joblib')


def test():
    # Load holdout test data
    holdout_data = pd.read_csv('test30.csv')
    holdout_X = holdout_data.drop('churned', 1)
    y = holdout_data['churned']

    # Load model
    model = joblib.load("model_submit1.joblib")

    # Predict with trained model
    predicted_y = model.predict(holdout_X)
    # predicted_y = model.predict_after_elimination(holdout_X)
    print("The model's F1-score on holdout set is", f1_score(y, predicted_y))

    # Display confusion matrix (print)
    (_0_0, _0_1), (_1_0, _1_1) = confusion_matrix(y, predicted_y)
    print(f'The confusion matrix is: 1-1 ({_1_1}), 0-0 ({_0_0}), 1-0 ({_1_0}), 0-1 ({_0_1})')


def split_csv():
    data = pd.read_csv('train.csv.gz')
    indeces = list(range(len(data)))
    test_indeces = random.choices(indeces, k=int(0.3 * len(indeces)))
    train_indeces = list(set(indeces) - set(test_indeces))
    data_train = data.iloc[train_indeces]
    data_train.to_csv('train70.csv', index=False, header=True)
    data_test = data.iloc[test_indeces]
    data_test.to_csv('test30.csv', index=False, header=True)


if __name__ == '__main__':
    split_csv()
    train()
    test()
