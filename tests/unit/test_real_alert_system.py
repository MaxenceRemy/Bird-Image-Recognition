import os
from monitoring.alert_system import AlertSystem

# Configuration des variables d'environnement pour le test
os.environ['ALERT_EMAIL'] = 'yoniedery26@gmail.com'
os.environ['EMAIL_PASSWORD'] = 'qzsh hkpz ytaa oevw'
os.environ['RECIPIENT_EMAIL'] = 'yoniedery26@gmail.com'

alert_system = AlertSystem()
result = alert_system.send_alert("Test Subject", "Test Message")

print("Email sent:", result)

# Nettoyage des variables d'environnement après le test
del os.environ['ALERT_EMAIL']
del os.environ['EMAIL_PASSWORD']
del os.environ['RECIPIENT_EMAIL']
