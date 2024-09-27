import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ChurnPredictor:
    def __init__(self):
        self.data = pd.read_csv("data/churn.csv")
        self.data = pd.get_dummies(self.data, columns=['Contract_Type'], drop_first=True)
        self.X = self.data.drop(columns=['Customer_ID', 'Churn_Flag'])
        self.y = self.data['Churn_Flag']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, new_data):
        new_data = pd.get_dummies(new_data, columns=['Contract_Type'], drop_first=True)
        missing_cols = set(self.X.columns) - set(new_data.columns)
        for col in missing_cols:
            new_data[col] = 0 
        new_data = new_data[self.X.columns]

        return self.model.predict_proba(new_data)[:, 1]

    def customer_retention_rate(self):
        retained = self.data['Churn_Flag'].value_counts().get(0, 0)
        total = len(self.data)
        return (retained / total) * 100

predictor = ChurnPredictor()
predictor.train_model()
new_customer_data = pd.DataFrame({'Monthly_Charges': [30], 'Tenure': [12], 'Contract_Type': ['One Year']})
probability_of_churn = predictor.predict(new_customer_data)
print('Churn Probability:',probability_of_churn)
print('Customer Retention Rate:',predictor.customer_retention_rate(),"%")
