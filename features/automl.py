import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

class ChurnPredictor:
    def __init__(self):
        self.data = pd.read_csv("data/churn.csv")
        self.data = pd.get_dummies(self.data, columns=['Contract_Type'], drop_first=True)
        
        self.X = self.data.drop(columns=['Customer_ID', 'Churn_Flag'])
        self.y = self.data['Churn_Flag']

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        models = {
            'Logistic Regression': (LogisticRegression(max_iter=100), {}),
            'Random Forest': (RandomForestClassifier(random_state=42), {
                'n_estimators': [10, 50, 100],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }),
            'XGBoost': (XGBClassifier( eval_metric='logloss'), {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            })
        }

        best_model = None
        best_accuracy = 0
        best_precision=0
        best_recall=0

        for model_name, (model, params) in models.items():
    # Perform hyperparameter tuning if parameters are provided
            if params:
                grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

    # Fit the model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

    # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

    # Update best model based on accuracy, then precision, then recall
            if accuracy > best_accuracy or (accuracy == best_accuracy and precision > best_precision) or (accuracy == best_accuracy and precision == best_precision and recall > best_recall):
                best_model = model
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall

# Final output for the best model
        if best_model is not None:
            print(f"Best Model: {best_model}")
            print(f"Best Accuracy: {best_accuracy:.2f}")
            print(f"Best Precision: {best_precision:.2f}")
            print(f"Best Recall: {best_recall:.2f}")



predictor = ChurnPredictor()
predictor.train_models()
