from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import time, io, os, time, sys, natsort, random, math
from PIL import Image
import base64,cv2
import numpy as np

# import library 
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

app = Flask(__name__)
# SocketIO는 ‘app’에 적용되고 있으며 나중에 애플리케이션을 실행할 때 앱 대신 socketio를 사용할 수 있도록 socketio 변수에 저장된다.
socketio = SocketIO(app,cors_allowed_origins='*' )

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4), # dot color
                             mp_drawing.DrawingSpec(color = (121, 44, 250), thickness = 2, circle_radius = 2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2, circle_radius = 4), # dot color
                             mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius = 2))

def extract_keypoints(results):
    lh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)    
    return np.concatenate([lh, rh])

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

actions = np.array(['None', '나', '달다', '도착', '돈', '맵다', '먼저', '무엇', '물음', 
                   '부탁', '사람', '시간', '얼마', '우리', '음식', '이거', '인기','있다',
                   '자리', '주문', '주세요', '짜다','책', '추천', '확인','제일'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss ='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights("C:/Users/woqja/Downloads/actionxhand_data25X90_0307_1423.h5")

font = ImageFont.truetype("fonts/HMFMMUEX.TTC", 10)
font2 = ImageFont.truetype("fonts/HMFMMUEX.TTC", 20)
blue_color = (255,0,0)



@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    # Take in base64 string and return PIL image
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    # convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

# 서버가 클라이언트가 보낸 메시지를 받는 방법, 클라이언트의 메시지를 확인하는 방법
# catch-frame 이벤트 핸들러 정의
# catch-frame 를 트리거 할 때 response_back 이벤트로 전송함 2번째 인자 data와 같이
@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  


global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]
global count
count = 0

# image 이벤트 핸들러 정의 클라이언트에서 image 이벤트 핸들러로 image data를 보냈으니 받는 것
@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    global count
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count = count+1
        cv2.putText(frame, 'Collecting frames for {}'.format(30-count), (30, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if (len(sequence) % 30 == 0):
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                u = np.bincount(predictions[-10:])
                b = u.argmax()
                if b == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                    
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] == 'None':
                                if count == 29:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                if(actions[np.argmax(res)] != sentence[-1]):
                                    if count ==29:
                                        sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

        cv2.rectangle(image, (0,0), (640, 20), (255, 255, 255), -1)
        image_modi = Image.fromarray(image)
        draw = ImageDraw.Draw(image_modi)
        draw.text((0,0), ' '.join(sentence), font=font2, fill=(0,0,0))
        image = np.array(image_modi)
        imgencode = cv2.imencode('.jpeg', image,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

        # base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData

        result_sentence = "제 일행이 먼저 도착 해 있어요."

        # emit the frame back
        socketio.emit('response_back', {comment:"안녕하세요"})
        if count ==29:
                count = 0


if __name__ == '__main__':
    socketio.run(app ,debug=True)