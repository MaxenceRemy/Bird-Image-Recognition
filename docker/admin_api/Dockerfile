FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
ADD admin_api.py .
ADD authorized_users.json .
EXPOSE 5100
CMD ["uvicorn", "admin_api:app", "--host", "0.0.0.0", "--port", "5100"]