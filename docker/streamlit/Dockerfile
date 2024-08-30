FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app/
ADD streamlit_docker.py .
EXPOSE 5300
CMD ["streamlit", "run", "streamlit_docker.py"]