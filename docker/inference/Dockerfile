FROM tensorflow/tensorflow:latest-gpu
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
ADD inference.py .
ADD load_image.jpg .
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "5500"]