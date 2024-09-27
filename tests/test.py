import unittest
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))

from churn import ChurnPredictor  # Adjusted import based on your structure

class TestChurnPredictor(unittest.TestCase):

    def test_prediction_accuracy(self):
        predictor = ChurnPredictor()
        predictor.train_model()

        # Check that predictions are valid probabilities (between 0 and 1)
        new_customer_data = pd.DataFrame({'Monthly_Charges': [30], 'Tenure': [12], 'Contract_Type': ['One Year']})
        probabilities = predictor.predict(new_customer_data)
        
        # Ensure predictions are between 0 and 1
        self.assertTrue(all(0 <= prob <= 1 for prob in probabilities))

    def test_retention_rate_calculation(self):
        predictor = ChurnPredictor()
        predictor.train_model()
        
        # Calculate retention rate
        retention_rate = predictor.customer_retention_rate()

        # Ensure retention rate is between 0% and 100%
        self.assertGreaterEqual(retention_rate, 0.0)
        self.assertLessEqual(retention_rate, 100.0)


if __name__ == '__main__':
    unittest.main()
