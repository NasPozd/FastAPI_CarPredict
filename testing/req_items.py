import requests
import pandas as pd
import numpy as np

df = pd.read_csv('file.csv').dropna()

items = []
for index, row in df.iterrows():
    item = {
        "name": row['name'],
        "year": row['year'],
        "selling_price": row['selling_price'],
        "km_driven": row['km_driven'],
        "fuel": row['fuel'],
        "seller_type": row['seller_type'],
        "transmission": row['transmission'],
        "owner": row['owner'],
        "mileage": row['mileage'],
        "engine": row['engine'],
        "max_power": row['max_power'],
        "torque": row['torque'],
        "seats": row['seats']
    }
    items.append(item)

payload = {
    "items": items
}

url = "http://localhost:8000/predict_items"
response = requests.post(url, json=payload)

if response.status_code == 200:
    print(response.text)
else:
    print("Error:", response.status_code, response.text)