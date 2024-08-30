FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
ADD preprocessing.py .
ADD SizeManager.py .
ADD UnderSampling.py .
ADD CleanDB.py .
ADD DatasetCorrection.py .
CMD ["uvicorn", "preprocessing:app", "--host", "0.0.0.0", "--port", "5500"]