import psutil
import time
import logging
import csv
from datetime import datetime
import os
import mlflow
import gpustat


class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
        self.logger.setLevel(logging.INFO)

        volume_path = 'volume_data'
        log_dir = os.path.join(volume_path, 'logs/system_monitor')
        os.makedirs(log_dir, exist_ok=True)

        #Récupération du nombre de gpus
        gpu_counts = len(gpustat.GPUStatCollection.new_query())

        self.csv_filename = os.path.join(log_dir, f'system_metrics_{datetime.now().strftime("%d%m%Y_%H%M")}.csv')
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            columns =  [
                    "Timestamp",
                    "CPU Usage",
                    "Memory Usage",
                    "Disk Usage",
                    "Network Sent",
                    "Network Recv",
                    "Swap Usage",
                    "Process Count",
                ]
            for i in range(gpu_counts):
                columns.append(f"GPU_{i} Usage")

            writer.writerow(columns)

    def get_metrics(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent
        net_io = psutil.net_io_counters()
        swap_usage = psutil.swap_memory().percent
        process_count = len(psutil.pids())
        gpus_usage = {}
        gpus_stats = gpustat.GPUStatCollection.new_query()
        for i, gpu in enumerate(gpus_stats.gpus):
            gpus_usage[f"gpu_{i}_usage"] = gpu.utilization

        metrics = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_sent": net_io.bytes_sent,
            "network_recv": net_io.bytes_recv,
            "swap_usage": swap_usage,
            "process_count": process_count,
        }

        metrics.update(gpus_usage)

        return metrics

    def log_metrics(self, metrics=None, timestamp=None):
        if metrics is None:
            metrics = self.get_metrics()
        if timestamp is None:
            timestamp = datetime.now()

        with open(self.csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp.strftime("%d-%m-%Y %H:%M:%S")] + list(metrics.values()))

        ts = int(timestamp.timestamp() * 1000)
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=0, timestamp=ts)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'enregistrement de la métrique {key} dans MLflow: {str(e)}")

    def monitor(self, duration=60, interval=5):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.log_metrics()
            time.sleep(interval)


if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor(duration=300, interval=5)
