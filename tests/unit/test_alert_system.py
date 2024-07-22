import unittest
from unittest.mock import patch, MagicMock
import os
from monitoring.alert_system import AlertSystem

class TestAlertSystem(unittest.TestCase):
    @patch('smtplib.SMTP_SSL')
    def test_send_alert(self, mock_smtp):
        # Configuration des variables d'environnement pour le test
        os.environ['ALERT_EMAIL'] = 'yoniedery26@gmail.com'
        os.environ['EMAIL_PASSWORD'] = 'qzsh hkpz ytaa oevw'
        os.environ['RECIPIENT_EMAIL'] = 'yoniedery26@gmail.com'

        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        alert_system = AlertSystem()
        result = alert_system.send_alert("Test Subject", "Test Message")

        self.assertTrue(result)
        mock_smtp_instance.login.assert_called_once_with('yoniedery26@gmail.com', 'qzsh hkpz ytaa oevw')
        mock_smtp_instance.sendmail.assert_called_once()

        # Nettoyage des variables d'environnement après le test
        del os.environ['ALERT_EMAIL']
        del os.environ['EMAIL_PASSWORD']
        del os.environ['RECIPIENT_EMAIL']

    @patch('smtplib.SMTP_SSL')
    def test_send_alert_failure(self, mock_smtp):
        mock_smtp.side_effect = Exception("Test error")

        alert_system = AlertSystem()
        result = alert_system.send_alert("Test Subject", "Test Message")

        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()