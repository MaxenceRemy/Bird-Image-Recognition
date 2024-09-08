FROM ubuntu:20.04
ADD requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app/
COPY streamlit.py .
COPY oiseau_cover.jpg .
COPY python_logo.png .
COPY docker_logo.png .
COPY mlflow_logo.png .
COPY tensorflow_logo.png .
COPY github_logo.png .
COPY architecture.svg .
COPY pipeline_user.svg .
COPY pipeline_mlops.svg .
COPY logo.jpeg .
EXPOSE 5300
CMD ["streamlit", "run", "streamlit.py"]