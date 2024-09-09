FROM tensorflow/tensorflow:latest-gpu
WORKDIR /home/app
ADD requirements.txt .
RUN apt-get update
RUN apt-get install python3-pip -y
RUN apt-get install -y git
RUN apt-get install -y supervisor
RUN pip3 install -r requirements.txt
ADD monitoring.py .
ADD system_monitor.py .
ADD drift_monitor.py .
ADD alert_system.py .
ADD supervisord.conf .
RUN mkdir -p /home/app/volume_data/logs
CMD ["/usr/bin/supervisord"]