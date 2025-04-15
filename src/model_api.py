import pandas as pd
import json
from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import os
import pickle
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import mle_project_challenge_2.create_model as create_model
from pydantic_models import InputFeatures

app = FastAPI()
model = None
model_features = None
demographics = None

def load_model():
    """
    train and dump the model
    only train if the model does not exist
    only load the model if it is not in memory
    in prod: only create the model once and pull it from s3/artifactory
    """
    global model, model_features, demographics, perf_metrics
    if model is None:
        if not os.path.exists('model/model.pkl'):
            print("Model not found, creating a new one...")
            create_model.main()
        print("Loading model...")
        model_path = os.path.join("model", "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # load demographics data if needed
    if demographics is None:
        print("Loading demographics...")
        demographics = pd.read_csv('data/zipcode_demographics.csv', dtype={'zipcode': str})
    if model_features is None:
        print("Loading model features...")
        model_features = json.load(open('model/model_features.json', 'r'))


"""
in the instructions it mentions a way to use a new model while api is deployed
this will train a new model, but in "production" we would simply
pull the latest model down from s3/artifactory and load it into memory
"""
@app.post("/refresh_model")
def refresh_model() -> dict:
    """
    trigger a reload of the model and demographics data while api is running
    """
    global model, demographics, model_features
    model = None
    demographics = None
    model_features = None
    return {"status": "Model and demographics data reloaded"}
    
@app.get("/")
def read_root() -> dict:
    """
    Root endpoint to check if the API is running.
    """
    return {'mle_project_challenge_2': 'House Price Prediction API'}


@app.get("/model_performance")
def get_model_performance() -> list[dict]:
    """
    Get the model performance metrics.
    Returns:
        List of dictionaries containing model performance metrics.
    """

    try:
        perf_metrics = json.load(open('model/model_perfs.json', 'r'))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model performance file not found")
    if len(perf_metrics) == 0:
        raise HTTPException(status_code=500, detail="Model not trained yet")
    
    return perf_metrics


"""
"Bonus: the basic model only uses a subset of the columns provided in the
house sales data.
Create an additional API endpoint where only the required features have
to be provided in order to get a prediction."

the use of the pydantic model meets the above requirement without the need 
for an additional endpoint because it only requires the features needed for the model
but allows for additional features to be passed in for completeness and forward compatibility
"""
@app.post("/houses/predict")
def predict(input_features: InputFeatures) -> dict:
    """
    Predict the house price based on input features.
    Args:
       **input_features**: InputFeatures object containing the features for prediction.
    """
    #check to see if the model is loaded, if not load it
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model could not be loaded: " + str(e))

    dem_subset = demographics.loc[demographics['zipcode'] == input_features.zipcode]

    input_df = pd.DataFrame([input_features.dict()])
    full_df = pd.merge(input_df, dem_subset, how="left", on="zipcode")

    if len(dem_subset) == 0:
        raise HTTPException(status_code=400, detail="Invalid zipcode")
    if len(dem_subset) > 1:
        raise HTTPException(status_code=400, detail="Multiple  Matches for zipcode found")

    X = full_df[model_features].to_numpy()
    prediction = model.predict(X)

    ret_dict = {
        "prediction": prediction[0],
        "input_features": full_df[model_features].to_dict(orient="records")[0],
    }
    
    # add the actual price if it is provided for evaluation
    if input_features.price is not None:
        ret_dict['actual'] = input_features.price

    return ret_dict
    

if __name__ == "__main__":
    # Load the model when the app starts. this could also be done in Dockerfile
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)