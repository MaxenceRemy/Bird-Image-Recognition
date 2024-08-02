from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
import hashlib
import os
import requests
import json

app = FastAPI()

volume_path = 'volume_data'

log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "user_api.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

@app.get("/")
def read_root():
    return {"Status": "OK"}
    

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
        try:

            content = await file.read()
            file_name = hashlib.sha256(content).hexdigest() + ".jpg"
            folder_path = os.path.join(volume_path, 'temp_images')
            os.makedirs(folder_path, exist_ok = True)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "wb") as image_file:
                image_file.write(content)

            response = requests.get(f'http://inference:5500/predict', params={'file_name': file_name})
            return response.json()
        
        except Exception as e:
            logging.error(f'Failed to communicate with the inference container: {e}')
            raise HTTPException(status_code=500, detail="Internal server error")

    
   

    
