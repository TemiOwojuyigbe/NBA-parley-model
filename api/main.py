import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('../models/wnba_model.pkl', 'rb') as f:
    model = pickle.load(f)

class ParlayRequest(BaseModel):
    player_name: str
    threshold: float
    pts_rolling_avg_5: float
    min_rolling_avg_5: float
    fga_rolling_avg_5: float
    is_home: int

@app.get('/')
def root():
    return {'message': 'WNBA Parlay Predictor API'}

@app.post('/predict')
def predict(request: ParlayRequest):
    features = [[
        request.pts_rolling_avg_5,
        request.min_rolling_avg_5,
        request.fga_rolling_avg_5,
        request.is_home
    ]]
    prob = model.predict_proba(features)[0][1]
    prediction = 'OVER' if prob > 0.5 else 'UNDER'
    return {
        'player': request.player_name,
        'threshold': request.threshold,
        'prediction': prediction,
        'probability': round(prob, 3)
    }