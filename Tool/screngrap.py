from PIL import Image  # 確保匯入正確的模組
import numpy as np
import win32gui
import win32ui
import win32con
import time
import cv2
import tensorflow as tf
import torchvision.transforms as transforms
class screngrap():
    def grap(Windowsname,number = 0,img2_return = False,d_width = 0 , d_height = 0 , d_top = 0 , d_left = 0):
        hwnd_target = win32gui.FindWindow(None, Windowsname)  # 獲取視窗句柄

        # 獲取視窗尺寸
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        top += 7 + d_top
        left += 10 + d_left
        w = d_width -7
        h = d_height -7

        # 設置前景窗口，等待穩定
        try:
            win32gui.SetForegroundWindow(hwnd_target)
        except:
            print("error")
        # time.sleep(1.0)

        # 截圖
        while True:

                hdesktop = win32gui.GetDesktopWindow()
                hwndDC = win32gui.GetWindowDC(hdesktop)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()

                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, int(w), int(h))
                saveDC.SelectObject(saveBitMap)

                saveDC.BitBlt((0, 0), (w, h), mfcDC, (int(left), int(top)), win32con.SRCCOPY)

                # 將位圖轉換為 NumPy 陣列
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (h, w, 4)
                img2 =  cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                # 清理資源
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hdesktop, hwndDC)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

                if img2_return:
                    pil_img = Image.fromarray(img2)
                    input_tensor = transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]
                    return input_tensor
                pil_img = Image.fromarray(img2)
                # pil_img.save(".\Temp\{}".format("Hollow Knight_" + str(number) + ".png"))
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

                
   
