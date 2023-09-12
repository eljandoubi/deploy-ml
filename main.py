# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
import pandas as pd
import joblib

class features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        json_schema_extra = [{

                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"

        },
        
        { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
            }
        ]

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def greetings():
    return "Welcome to Census Bureau Classifier API"


model, encoder, lb = joblib.load("./model/transformers.pkl")
cat_features = [f for (f,t) in features.__annotations__.items() if t==str]

# Use POST action to send data to the server
@app.post("/invocations")
async def predictor(body: features):
    
    data = pd.DataFrame(body.__dict__,[0])
    
    data, * _ = process_data(data, categorical_features=cat_features,
                                        training=False, encoder=encoder)
    
    pred = inference(model,data)
    
    return lb.inverse_transform(pred)[0]
