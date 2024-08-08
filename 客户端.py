# client.py
import requests
import json

# 定义要发送的JSON数据
# 解析JSON数据
# with open('1.json', 'r') as file:
#     data = json.load(file)
file="1.json"
# 将数据发送到服务端
response = requests.post('http://127.0.0.1:5000/process', files={'file': file})

# 打印服务端的响应
if response.status_code == 200:
    print("Response from server:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Failed to send data to server. Status code: {response.status_code}")
    print("Response:", response.text)
