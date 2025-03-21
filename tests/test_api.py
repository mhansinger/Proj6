from fastapi.testclient import TestClient 
import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

from main import app 

# Create a test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root GET endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello and welcome!"

def test_predict_endpoint():
    """Test the predict POST endpoint with valid data"""
    sample_input = {
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
    
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]  # Expected output

def test_invalid_request():
    invalid_data = {"age": -1, "workclass": "State-gov"}  # Age is negative and missing data
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
