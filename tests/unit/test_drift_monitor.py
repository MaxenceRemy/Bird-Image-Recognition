import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from monitoring.drift_monitor import DriftMonitor
import os
import json

class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        self.test_log_file = 'test_performance_logs.csv'
        self.test_initial_counts = {'class1': 100, 'class2': 100}
        
        # Create a test CSV file
        test_data = pd.DataFrame({
            'date': ['2023-01-01'] * 220,
            'predicted_class': ['class1'] * 110 + ['class2'] * 110,
            'confidence': [0.9] * 220
        })
        test_data.to_csv(self.test_log_file, index=False)
        
        # Create a test initial_class_counts.json
        with open('initial_class_counts.json', 'w') as f:
            json.dump(self.test_initial_counts, f)

    def tearDown(self):
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        if os.path.exists('initial_class_counts.json'):
            os.remove('initial_class_counts.json')

    def test_check_drift(self):
        monitor = DriftMonitor(self.test_log_file)
        drift_detected, reasons = monitor.check_drift()
        
        self.assertTrue(drift_detected)
        self.assertIn("Class class1 increased by more than 5%", reasons[0])
        self.assertIn("Class class2 increased by more than 5%", reasons[1])

    def test_check_drift_no_data(self):
        empty_log_file = 'empty_log.csv'
        open(empty_log_file, 'w').close()  # Create an empty file
        
        monitor = DriftMonitor(empty_log_file)
        drift_detected, reason = monitor.check_drift()
        
        self.assertFalse(drift_detected)
        self.assertEqual(reason, "Pas assez de données pour détecter un drift")
        
        os.remove(empty_log_file)

if __name__ == '__main__':
    unittest.main()