import unittest
import random
from app.models.predictClass import predictClass
from monitoring.performance_tracker import PerformanceTracker

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.classifier = predictClass()
        self.tracker = PerformanceTracker()
        self.classes = ['class1', 'class2', 'class3']  # Remplacez par vos vraies classes

    def test_pipeline(self):
        # Simuler 100 prédictions
        for _ in range(100):
            predicted_class = random.choice(self.classes)
            true_class = random.choice(self.classes)
            confidence = random.uniform(0.5, 1.0)
            self.tracker.log_prediction(predicted_class, confidence, true_class)

        # Vérifier que les prédictions ont été enregistrées
        overall_accuracy, class_accuracies = self.tracker.get_performance_metrics()
        self.assertIsInstance(overall_accuracy, float)
        self.assertIsInstance(class_accuracies, dict)
        self.assertTrue(0 <= overall_accuracy <= 1)
        
        # Vérifier que toutes les classes sont présentes dans les résultats
        for class_name in self.classes:
            self.assertIn(class_name, class_accuracies)

if __name__ == '__main__':
    unittest.main()