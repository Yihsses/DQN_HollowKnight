import time
from Tool.screngrap import screngrap
s = screngrap
number = 0
while True:
    print("列印成功")
    time.sleep(0.2)
    screngrap.grap("Hollow Knight" , number)
    number += 1
    