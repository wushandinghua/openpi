import requests



# 服务端的URL

url = 'http://192.168.1.116:5000/message'



# 要发送的消息

message = {'message': 'Hello, Flask Server!'}



# 发送POST请求

response = requests.post(url, json=message)



# 打印服务端的响应

print(response)
#print(response.json())
