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
        top += 32
        left += 10
        w = right - left-7
        h = bot - top-7

        # 設置前景窗口，等待穩定
        try:
            win32gui.SetForegroundWindow(hwnd_target)
        except:
            print("error")
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
        resized_img.save("test.png")
        # 轉回 NumPy 陣列
        resized_img_np = np.array(resized_img)

        return resized_img_np
    def grap_hp(Windowsname):
        """
        截取指定視窗內的一部分 (假設為 HP 區域) 並保存為圖片，無需調整大小。
        
        :param Windowsname: 要截取的視窗名稱
        """
        hwnd_target = win32gui.FindWindow(None, Windowsname)  # 獲取視窗句柄

        if not hwnd_target:
            raise ValueError(f"找不到名稱為 '{Windowsname}' 的視窗")

        # 獲取視窗尺寸
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        top += 80  # 調整區域起點
        left += 180
        w = 310  # 寬度
        h = 50   # 高度

        # 設置前景窗口，等待穩定
        try:
            win32gui.SetForegroundWindow(hwnd_target)
        except Exception as e:
            print("無法設置前景窗口:", e)
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

        # 保存圖片
        pil_img = Image.fromarray(img)
        pil_img.save("hp.png")
        print("HP 圖片已保存為 'hp.png'")
        resized_img_np = np.array(pil_img)
        return np.array(resized_img_np)
