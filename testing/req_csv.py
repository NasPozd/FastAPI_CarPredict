import requests
import pandas as pd
from io import StringIO

url = "http://localhost:8000/predict_csv"
files = {'file': open('file.csv', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.content.decode('utf-8')), header=None)
    print(df.head())
else:
    print("Error:", response.status_code, response.text)