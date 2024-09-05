FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app/
COPY streamlit_docker.py .
COPY architecture.svg .
COPY oiseau_cover.jpg .
COPY pipeline_user.svg .
COPY pipeline_mlops.svg .
COPY logo.jpeg .
EXPOSE 5300
CMD ["streamlit", "run", "streamlit_docker.py"]