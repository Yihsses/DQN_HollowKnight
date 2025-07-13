from PIL import Image  # 確保匯入正確的模組
import numpy as np
import win32gui
import win32ui
import win32con
import time
import cv2
import tensorflow as tf

class screngrap():
    def grap(Windowsname,number = 0,img2_return = False):
        hwnd_target = win32gui.FindWindow(None, Windowsname)  # 獲取視窗句柄

        # 獲取視窗尺寸
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        top += 32+60
        left += 10
        w = right - left-7
        h = bot - top-7-70

        # 設置前景窗口，等待穩定
        try:
            win32gui.SetForegroundWindow(hwnd_target)
        except:
            print("error")
        # time.sleep(1.0)

        # 截圖
        while True:
            try:
                hdesktop = win32gui.GetDesktopWindow()
                hwndDC = win32gui.GetWindowDC(hdesktop)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()

                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
                saveDC.SelectObject(saveBitMap)

                saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)

                # 將位圖轉換為 NumPy 陣列
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (h, w, 4)
                img2 =  cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                # 清理資源
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hdesktop, hwndDC)

                if(img2_return):return Image.fromarray(img2)
                pil_img = Image.fromarray(img2)
                # pil_img.save(".\YOLO DATA\{}".format("Hollow Knight_" + str(number) + ".png"))
                pil_img.save("YOLO.png")
                resized_img = cv2.resize(img, (160, 160))
                # cv2.imwrite(".\ELDING BOT\YOLO DATA\{}".format("Hollow Knight_" + str(number)) , resized_img)
                resized_img = np.array(resized_img)
                # pil_img = Image.fromarray(img)
                # pil_img = pil_img.convert('L') 
            
                # resized_img = pil_img.resize((400, 200), Image.Resampling.LANCZOS)
                # img_np = np.array(pil_img) 
                # resized_img = cv2.resize(img_np, (400, 200), interpolation=cv2.INTER_LANCZOS4)
                # cv2.imwrite("output_image.png", resized_img)  # 儲存為 PNG 格式
                # 轉回 NumPy 陣列
                # resized_img_np = np.array(resized_img)
                
                return resized_img
            except:
                print("error")
                
   
