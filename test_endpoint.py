import requests
import pandas as pd

url = 'http://localhost:8000/houses/predict'

data = pd.read_csv('src/data/future_unseen_examples.csv')
bodies = data.to_dict(orient='records')

for body in bodies:
    response = requests.post(url, json=body)
    if response.status_code == 200:
        print("Response from API:", response.json()['prediction'])
        print("Input features:", response.json()['input_features'], "\n")
    else:
        print(f"Failed to get a valid response. Status code: {response.status_code}, Response: {response.text}\n")