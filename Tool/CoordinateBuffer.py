import socket

class CoordinateClient:
    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port

    def parse_data(self, data: str):
        """解析 socket 傳來的字串資料，轉成 dict"""
        data = data.decode().replace("\n", '')
        parts = data.split("/")  # 以斜線切分

        result = []
        try:
            for i in range(0, len(parts)-1, 2):
                hp = int(parts[i])
                coords = parts[i+1].strip("()").split(",")
                x = float(coords[0])
                y = float(coords[1])
                result.append({
                    "hp": hp,
                    "x": x,
                    "y": y
                })

            if len(parts) > 4:
                result.append({"mp": parts[4]})
        except Exception as e:
            print("解析錯誤:", e, "data:", data)

        return result

    def get_coordinates(self):
        """直接連線一次，取得一筆座標"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                data = s.recv(1024)
                if not data:
                    return None
                return self.parse_data(data)
        except Exception as e:
            print("Socket 連線錯誤:", e)
            return None
