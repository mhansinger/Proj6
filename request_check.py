import json
import requests

data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Masters",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "Canada"
    }

response = requests.post(
    "https://proj6-deploy-ml.onrender.com/predict", data=json.dumps(data))

print(response.status_code)
print(response.json())