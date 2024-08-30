from fastapi import FastAPI, HTTPException
import logging
import requests
import json
import os
import time
import datetime
import smtplib, ssl

app = FastAPI()

volume_path = 'volume_data'

log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "training.log"), level=logging.INFO, 
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
    

@app.get("/")
def read_root():
    return {"Status": "OK"}
    

@app.get("/train")
async def train():
        
    try:

        folder_path = os.path.join(volume_path, 'dataset_clean')
        file_path = os.path.join(folder_path, "face.jpg")
        with open(file_path, 'r'):
            pass
       
        
        time.sleep(5)
        archive_folder = os.path.join(volume_path, "models_archives")
        os.makedirs(archive_folder, exist_ok = True)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%d-%m-%Y_%H-%M-%S") + ".h5"
        model_file = os.path.join(archive_folder, date_time)
        with open(model_file, 'w') as f:
            f.write("This is a fake h5 model, that doesn't work obviously.")
        fake_training_results = f"""

        Training date and time : {datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}
        -------------------------------

        Epoch 1/3
        -------------------------------
        Loss: 1.2564
        Accuracy: 65.24%
        Validation Loss: 1.1023
        Validation Accuracy: 68.78%

        Epoch 2/3
        -------------------------------
        Loss: 0.9743
        Accuracy: 72.89%
        Validation Loss: 0.8934
        Validation Accuracy: 74.56%

        Epoch 3/3
        -------------------------------
        Loss: 0.8432
        Accuracy: 76.45%
        Validation Loss: 0.7841
        Validation Accuracy: 77.89%

        Final Metrics
        -------------------------------
        Final Training Loss: 0.8432
        Final Training Accuracy: 76.45%
        Final Validation Loss: 0.7841
        Final Validation Accuracy: 77.89%
        """
        
        with open(training_logs_file, 'w') as f:
            f.write(fake_training_results)

        
        send_email("Training just ended, please go the admin API and get the training results on the --result-- endpoint.")

    
    except Exception as e:
        logging.error(f'Failed to load the dataset and/or do the training: {e}')
        error_message = f"""
        Date and time : {datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}
        -------------------------------

        Something went wrong with the preprocessing and/or training, 
        please look at the logs of --training-- and --preprocessing--
        """
        with open(training_logs_file, 'w') as f:
            f.write(error_message)
        send_email(error_message)