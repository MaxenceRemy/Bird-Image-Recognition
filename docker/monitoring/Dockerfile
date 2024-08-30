FROM tensorflow/tensorflow:latest-gpu
WORKDIR /home/app
ADD requirements.txt .
RUN apt-get update 
RUN apt-get install python3-pip -y 
RUN apt-get install -y git 
RUN apt-get install -y cron
RUN pip3 install -r requirements.txt 
ADD monitoring.py .
ADD system_monitor.py .
ADD alert_system.py .
ADD drift_monitor.py .
RUN touch /var/log/cron.log
RUN (crontab -l 2>/dev/null; echo "0 0 * * * python3 /home/app/drift_monitor.py >> /var/log/cron.log 2>&1") | crontab -
CMD cron && tail -f /var/log/cron.log & python3 monitoring.py
