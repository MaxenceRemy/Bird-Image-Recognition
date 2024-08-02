FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
ADD inference.py .
RUN mkdir -p volume_data/prod_weights
ADD main_model.h5 .
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "5500"]