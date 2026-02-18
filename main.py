from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('digits_model.joblib')

class Features(BaseModel):
    features : list


@app.get('/')
def welcome():
    return {'message': 'Welcome to Digits classification', 'status': 200}


@app.post('/predict')
def predict(data: Features):
    try:
        x = np.array(data.features).reshape(1,-1)
        predict_value = model.predict(x)[0]
        return {'predict_value': int(predict_value), 'status': 200}
    except Exception as e:
        print('Exception: ', str(e))
        return HTTPException(status_code=500, detail=str(e))
