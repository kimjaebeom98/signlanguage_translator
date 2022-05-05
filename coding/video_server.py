#video_server.py
from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


# 웹소켓을 통해 메시지(dataURL)를 받으면 opencv로 읽을수 있는 형태로 변환
class SimpleEcho(WebSocket):
    def handle(self):
        msg = self.data
        img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('image', img)
        cv2.waitKey(1)
        

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')

# 웹소켓 서버를 생성합니다. localhost 부분은 ip주소(예를들면 192.168.0.1), 3000은 port 번호입니다.

server = WebSocketServer('localhost', 3000, SimpleEcho)
server.serve_forever()