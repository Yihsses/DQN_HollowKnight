import socket
import threading
from Tool.CoordinateBuffer import CoordinateClient
import time
# 'hp': 8, 'x': 15.3, 'y': 28.4 左邊
# 'hp': 3, 'x': 37.6, 'y': 37.9} 右上
print( 4 // 5)
while True:
    try:
        cb = CoordinateClient()
        print(cb.get_coordinates())
        time.sleep(1)
    except:
        print()
