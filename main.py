from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
from ml.data import process_data
from ml.model import inference
import pandas as pd
import pdb


# Initialize API object
app = FastAPI()

# Load model and stuff
model_path = "model/rf_model.pkl"
rf_model = pickle.load(open(model_path, "rb"))

encoder_path = "model/encoder.pkl"
encoder = pickle.load(open(encoder_path, "rb"))

lb_path = "model/lb.pkl"
lb = pickle.load(open(lb_path, "rb"))


# Expected Input Data schema with PyDantic (excluding label)
class InputData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="Canada")


@app.get("/")
async def greeter() -> str:
    """GET method greeter

    Returns:
        str: _description_
    """
    return "Hello and welcome!"


@app.post("/predict")
async def predict(data: InputData):
    """POST method to predict income category

    Args:
        data (InputData): Input data as pydantic schema

    Returns:
        _type_: _description_
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    sample = {key.replace("_", "-"): [value] for key, value in data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(sample)
    X_processed, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    output = inference(rf_model, X_processed)[0]
    str_out = "<=50K" if output == 0 else ">50K"
    return {"prediction": str_out}

