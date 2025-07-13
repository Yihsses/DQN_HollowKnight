import socket
import threading
print( 4 // 5)
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect(('127.0.0.1', 5555))
#         while True:
#             try:
#                 data = s.recv(1024)
#                 if not data:
#                     break  # 對方關閉連線
#                 data = data.decode().replace("\n", '')
#                 parts = data.split("/")  # 先以斜線切開

#                 result = []

#                 for i in range(0, len(parts), 2):
#                     hp = int(parts[i])
#                     coords = parts[i+1].strip("()").split(",")
#                     x = float(coords[0])
#                     y = float(coords[1])
#                     result.append({
#                         "hp": hp,
#                         "x": x,
#                         "y": y
#                     })

#                 print(result)

#             except Exception as e:
#                 print("Error:", e)
                