from flask import Flask, render_template, Response
import cv2
#플라스크 객체 생성
app = Flask(__name__)
camera = cv2.VideoCapture(0)

# import library 
import natsort
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import sys
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import random
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

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
model.load_weights('C:/Users/MASTER/actionxhand_data25X90_0307_1423.h5')


font = ImageFont.truetype("fonts/HMFMMUEX.TTC", 10)
font2 = ImageFont.truetype("fonts/HMFMMUEX.TTC", 20)
blue_color = (255,0,0)
cap = cv2.VideoCapture(0)

def generate_frames():
    global cap
    flag = 0
    while (True):
        ret, img = cap.read()
        if not ret:
            break
        else:
            detector = cv2.CascadeClassifier('C:/Users/MASTER/Desktop/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            img = cv2.rectangle(img, (210,30), (410,210), blue_color, 2)
        
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if(x>=210 and x<=250 and y>=30 and y<=45):
                    if(x+w>=370 and x+w<=410 and y+h>=180 and y+h<=195 ):
                        flag = 1
            if (flag ==1):
                break

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img+ b'\r\n')
# 1. New detection variables

#사용자의 사용위치 찾기


    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    count = 0
    cap = cv2.VideoCapture(0)
# Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

        # Read feed
            ret, frame = cap.read()
            count = count+1
            cv2.putText(frame, 'Collecting frames for {}'.format(30-count), (30, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
        
        # Draw landmarks
            draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
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

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
        
            cv2.rectangle(image, (0,0), (640, 20), (255, 255, 255), -1)
            image_modi = Image.fromarray(image)
            draw = ImageDraw.Draw(image_modi)
            draw.text((0,0), ' '.join(sentence), font=font2, fill=(0,0,0))
            image = np.array(image_modi)
        # Show to screen
            ret, buffer = cv2.imencode('.jpg', image)
            img = buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img+ b'\r\n')
            if count ==29:
                count = 0

#만일 static폴더를 지정하고 싶다면
#app = Flask(__name, static_folder='./static/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype ='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)