from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('./digits_model.joblib')

class Features(BaseModel):
    features : List[List[float]]


@app.get('/health_check')
def welcome():
    print('Welcome to Digits classification')
    return {'message': 'Welcome to Digits classification', 'status': 200}


@app.post('/predict')
def predict(data: Features):
    try:
        x = np.array(data.features)
        predict_value = model.predict(x)
        return {'predict_value': predict_value.tolist(), 'status': 200}
    except Exception as e:
        print('Exception: ', str(e))
        return HTTPException(status_code=500, detail=str(e))
