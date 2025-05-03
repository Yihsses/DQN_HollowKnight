from PIL import Image  # 確保匯入正確的模組
import numpy as np
import win32gui
import win32ui
import win32con
import time

class screngrap():

    def grap(Windowsname):
        hwnd_target = win32gui.FindWindow(None, Windowsname)  # 獲取視窗句柄

        # 獲取視窗尺寸
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        w = right - left
        h = bot - top

        # 設置前景窗口，等待穩定
        win32gui.SetForegroundWindow(hwnd_target)
        time.sleep(1.0)

        # 截圖
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

        # 清理資源
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)

        # 使用 PIL 調整大小到 1280x720
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize((1280, 720), Image.Resampling.LANCZOS)

        # 轉回 NumPy 陣列
        resized_img_np = np.array(resized_img)

        return resized_img_np
