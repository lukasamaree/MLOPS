import requests

url = "http://localhost:8000/predict"

sample_input = {
    "features": [
        4.2, 85, 45.5, 3100, 15000, 15, 520, 102.5, 1.5, 4.9, 48, 35,
        False, True, False,
        True, False, False, True, False, False,
        True, False
    ]
}

response = requests.post(url, json=sample_input)
print("Server response:", response.json())