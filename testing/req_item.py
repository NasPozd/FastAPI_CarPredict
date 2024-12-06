import pandas as pd
import requests

df = pd.read_csv('file.csv')
url = "http://localhost:8000/predict_item"
response = requests.post(url, json=df.sample(n=1).iloc[0].to_dict())

if response.status_code == 200:
    print("Predicted price:", response.json())
else:
    print("Error:", response.json())