from fastapi import FastAPI, HTTPException
import logging
import requests
import json
import os

app = FastAPI()

volume_path = 'volume_data'

log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "admin_api.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

@app.get("/")
def read_root():
    return {"Status": "OK"}
    

@app.get("/train")
async def train():
        
        try:
            response = requests.get(f'http://preprocessing:5500/preprocess')
            return response.json()
        
        except requests.RequestException as e:
            logging.error(f'Failed to communicate with the preprocessing container: {e}')
            raise HTTPException(status_code=500, detail="Internal server error")
        
@app.get("/results")
async def train():
        
        try:
            file_path = os.path.join(log_folder, "training_results.txt")
            with open(file_path, 'r') as file:
                results = file.read()
                
            return {"Training results :": results}
        
        except requests.RequestException as e:
            logging.error(f'Failed to open the training results file: {e}')
            raise HTTPException(status_code=500, detail="Internal server error")