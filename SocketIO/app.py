from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import time, io, os, time, sys, natsort, random, math, joblib
import pandas as pd
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
model.load_weights("C:/Users/owner/Desktop/actionxhand_data25X90_0307_1423.h5")

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

my_dict ={"None":0, "나":1, "달다":2, "도착":3, "돈":4, "맵다" :5, "먼저": 6,
         "무엇":7, "물음":8, "부탁":9, "사람":10, "시간":11, "얼마":12, "우리":13, 
          "음식":14, "이거":15, "인기":16, "있다":17, "자리":18, "주문":19, "주세요":20, 
          "짜다":21, "책":22, "추천":23, "확인":24, "제일":25}

def make_word_df(word0, word1, word2, word3, word4):
    info = [[word0, word1, word2, word3, word4]]
    df = pd.DataFrame(info, columns = ['target0', 'target1', 'target2', 'target3', 'target4'])
    return df

def get_key(val):
    for key, value in my_dict.items():
         if val == key:
             return value
 
    return "There is no such Key"

def make_num_df(input_1):
    num_oflist = []
    for i in input_1.columns:
        num_oflist.append(get_key(input_1[i].values))
    input2 = make_word_df(num_oflist[0], num_oflist[1], num_oflist[2], num_oflist[3], num_oflist[4])
    return input2

# 서버가 클라이언트가 보낸 메시지를 받는 방법, 클라이언트의 메시지를 확인하는 방법
# catch-frame 이벤트 핸들러 정의
# catch-frame 를 트리거 할 때 response_back 이벤트로 전송함 2번째 인자 data와 같이
@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  

global count, sequence, sentence, predictions
sequence = []
sentence = []
predictions = []
count = 0
rlf = RandomForestClassifier()
rlf = joblib.load("문장생성 모델") ##모델을 만들면 추가 할것임
le = LabelEncoder()

# image 이벤트 핸들러 정의 클라이언트에서 image 이벤트 핸들러로 image data를 보냈으니 받는 것
@socketio.on('image')
def image(data_image):
    frame = (readb64(data_image))
    threshold = 0.7
    global count, sequence, sentence, predictions
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count = count+1
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

        ##문장 생성 모델을 load
        if count == 29:
            count = 0
            if(len(sentence) != 0) :
                if(len(sentence) == 5):
                    input = make_word_df(sentence[0], sentence[1], sentence[2], sentence[3], sentence[4])
                    word_input = make_num_df(input)
                    y_prd = rlf.predict(word_input)
                    predict_sentence = np.array2string(le.inverse_transform(y_prd))
                    ##outputSentence에는 문장이 들어있다. 이를 이제 이미지화해서 문장을 출력하자
                    img = np.full((200, 300, 3), (255, 255, 255), np.unit8)
                    img = Image.fromarray(img)
                
                    draw = ImageDraw.Draw(img)
                    draw.text((60, 80), predict_sentence, font = font2, fill=(0,0,0))
                    image_send = np.array(img)
                    imgencode = cv2.imencode('.jpeg', image_send,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

                    stringData = base64.b64encode(imgencode).decode('utf-8')
                    b64_src = 'data:image/jpeg;base64,'
                    stringData = b64_src + stringData

                    # emit the frame back
                    emit('response_back', img)
                else:
                    predict_word = sentence[-1]
                    img = np.full((200, 300, 3), (255, 255, 255), np.uint8)
                    img = Image.fromarray(img)
                        
                    draw = ImageDraw.Draw(img)
                    draw.text((60, 80), predict_word, font = font2, fill=(0,0,0))
                    image_send = np.array(img)
                    imgencode = cv2.imencode('.jpeg', image_send,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

                    stringData = base64.b64encode(imgencode).decode('utf-8')
                    b64_src = 'data:image/jpeg;base64,'
                    stringData = b64_src + stringData

                    # emit the frame back
                    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app ,debug=True)