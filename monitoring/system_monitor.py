import psutil
import time
import logging
import csv
from datetime import datetime
import os
import mlflow

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger('system_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Créer un gestionnaire de fichiers pour les logs
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f'system_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Définir le format du log
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Ajouter le gestionnaire au logger
        self.logger.addHandler(file_handler)
        
        # Créer un fichier CSV pour les métriques
        self.csv_filename = os.path.join(log_dir, f'system_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'CPU Usage', 'Memory Usage', 'Disk Usage', 'Network Sent', 'Network Recv', 'Swap Usage', 'Process Count'])

    def get_metrics(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        net_io = psutil.net_io_counters()
        swap_usage = psutil.swap_memory().percent
        process_count = len(psutil.pids())
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'network_sent': net_io.bytes_sent,
            'network_recv': net_io.bytes_recv,
            'swap_usage': swap_usage,
            'process_count': process_count
        }

    def log_metrics(self):
        metrics = self.get_metrics()
        log_message = f"CPU: {metrics['cpu_usage']}% | Memory: {metrics['memory_usage']}% | Disk: {metrics['disk_usage']}% | " \
                      f"Net Sent: {metrics['network_sent']} | Net Recv: {metrics['network_recv']} | " \
                      f"Swap: {metrics['swap_usage']}% | Processes: {metrics['process_count']}"
        
        self.logger.info(log_message)
        
        # Écrire dans le fichier CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + list(metrics.values()))

        # Log des métriques dans MLflow
        mlflow.log_metrics({
            'cpu_usage': metrics['cpu_usage'],
            'memory_usage': metrics['memory_usage'],
            'disk_usage': metrics['disk_usage'],
            'network_sent': metrics['network_sent'],
            'network_recv': metrics['network_recv'],
            'swap_usage': metrics['swap_usage'],
            'process_count': metrics['process_count']
        })

    def monitor(self, duration=60, interval=5):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.log_metrics()
            time.sleep(interval)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor(duration=300, interval=5)  # Surveiller pendant 5 minutes, enregistrer toutes les 5 secondes