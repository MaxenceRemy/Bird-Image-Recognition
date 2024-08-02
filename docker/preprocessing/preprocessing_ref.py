from fastapi import FastAPI, HTTPException, BackgroundTasks
import logging
import requests
import json
import os
from PIL import Image
import datetime
import smtplib, ssl

app = FastAPI()

volume_path = 'volume_data'

log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "preprocessing.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

training_logs_file = os.path.join(log_folder, "training_results.txt")

def send_email(message):

    login__email = "raspberrypisender6@gmail.com"
    login_password = "seet pbgx mmme uyzz"

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(login__email, login_password)
        server.sendmail(login__email, "maxenceremy1@gmail.com", message)

def preprocessing():
    try:
        dataset = requests.get("https://thispersonsnotexist.com")
        
        dataset_raw_path = os.path.join(volume_path, 'dataset_raw')
        dataset_clean_path = os.path.join(volume_path, 'dataset_clean')
        os.makedirs(dataset_raw_path, exist_ok = True)
        os.makedirs(dataset_clean_path, exist_ok = True)

        raw_image_path = os.path.join(dataset_raw_path, "face.jpg")
        with open(raw_image_path, 'wb') as file:
            file.write(dataset.content)
        clean_image_path = os.path.join(dataset_clean_path, "face.jpg")
        with Image.open(raw_image_path) as image:
            bw_img = image.convert('L')
            bw_img.save(clean_image_path)

        
        requests.get(f'http://training:5500/train')

        

    except Exception as e:
        logging.error(f'Failed to preprocess the dataset: {e}')
        error_message = f"""
        Date and time : {datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}
        -------------------------------

        Something went wrong with the preprocessing and/or training, 
        please look at the logs of --training-- and --preprocessing--
        """
        with open(training_logs_file, 'w') as f:
            f.write(error_message)
        send_email(error_message)

@app.get("/")
def read_root():
    return {"Status": "OK"}
    

@app.get("/preprocess")
async def preprocess(background_tasks: BackgroundTasks):
        
        
    background_tasks.add_task(preprocessing)
    return {"Status": "Preprocessing started. The training will be automtically launched. You'll get an e-mail when training is over.\
            Then, please go to the --result-- API endpoint to get the training results. The model will be saved in the models archives."}
        
        