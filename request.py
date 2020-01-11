import requests

url = 'http://localhost:5000/api'
r = requests.get(url,json={'exp':1.8,})
print(r.json())