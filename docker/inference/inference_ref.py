from fastapi import FastAPI, HTTPException
import logging
import os

app = FastAPI()

volume_path = 'volume_data'

log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "inference.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

@app.get("/")
def read_root():
    return {"Status": "OK"}

    

@app.get("/predict")
async def predict(file_name: str):
    try:

        folder_path = os.path.join(volume_path, 'temp_images')
        os.makedirs(folder_path, exist_ok = True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r'):
            pass
        # code de prédiction
        os.remove(file_path)
        return {"Prediction label": "COOL BIRD", "Score": "0.98"}
    
    except Exception as e:
        logging.error(f'Failed to open the image and/or do the inference: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")

    
