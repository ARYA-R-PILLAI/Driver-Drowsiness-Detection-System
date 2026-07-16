import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import time
import math
from playsound import playsound
import threading
from datetime import datetime

# ==========================================
# DRIVER DROWSINESS DETECTION SYSTEM
# WEB VERSION
# ==========================================

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

EAR_THRESHOLD = 0.45
MAR_THRESHOLD = 0.60

DROWSY_TIME = 1.0
YAWN_TIME = 1.0

closed_start = None
yawn_start = None

alarm_playing = False

eye_closed = False
mouth_open = False

blink_count = 0
yawn_count = 0

fps = 0
prev_time = time.time()

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


def distance(p1,p2):

    return math.sqrt(

        (p1.x-p2.x)**2+

        (p1.y-p2.y)**2

    )


def play_alarm():

    global alarm_playing

    playsound("alarm.wav")

    alarm_playing=False


def generate_frames():

    global closed_start
    global yawn_start
    global alarm_playing
    global eye_closed
    global mouth_open
    global blink_count
    global yawn_count
    global fps
    global prev_time
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        status = "No Face"
        color = (255,255,255)

        ear = 0
        mar = 0

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape

            # ==========================
            # LEFT EYE
            # ==========================

            lp1 = landmarks[33]
            lp2 = landmarks[160]
            lp3 = landmarks[158]
            lp4 = landmarks[133]
            lp5 = landmarks[153]
            lp6 = landmarks[144]

            leftEAR = (
                distance(lp2, lp5) +
                distance(lp3, lp6)
            ) / (2 * distance(lp1, lp4))

            # ==========================
            # RIGHT EYE
            # ==========================

            rp1 = landmarks[362]
            rp2 = landmarks[385]
            rp3 = landmarks[387]
            rp4 = landmarks[263]
            rp5 = landmarks[373]
            rp6 = landmarks[380]

            rightEAR = (
                distance(rp2, rp5) +
                distance(rp3, rp6)
            ) / (2 * distance(rp1, rp4))

            ear = (leftEAR + rightEAR) / 2            # ==========================
            # MOUTH
            # ==========================

            top = landmarks[TOP_LIP]
            bottom = landmarks[BOTTOM_LIP]
            left = landmarks[LEFT_MOUTH]
            right = landmarks[RIGHT_MOUTH]

            mouth_vertical = distance(top, bottom)
            mouth_horizontal = distance(left, right)

            mar = mouth_vertical / mouth_horizontal

            # ==========================
            # DRAW EYES
            # ==========================

            eye_points = LEFT_EYE + RIGHT_EYE

            for point in eye_points:

                x = int(landmarks[point].x * w)
                y = int(landmarks[point].y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            # ==========================
            # DRAW MOUTH
            # ==========================

            mouth_points = [
                TOP_LIP,
                BOTTOM_LIP,
                LEFT_MOUTH,
                RIGHT_MOUTH
            ]

            for point in mouth_points:

                x = int(landmarks[point].x * w)
                y = int(landmarks[point].y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (255, 0, 255),
                    -1
                )            # ==========================
            # BLINK DETECTION
            # ==========================

            if ear < EAR_THRESHOLD:

                if not eye_closed:
                    eye_closed = True

                if closed_start is None:
                    closed_start = time.time()

                eye_elapsed = time.time() - closed_start

            else:

                if eye_closed:
                    blink_count += 1

                eye_closed = False
                closed_start = None
                eye_elapsed = 0

            # ==========================
            # YAWN DETECTION
            # ==========================

            if mar > MAR_THRESHOLD:

                if not mouth_open:
                    mouth_open = True

                if yawn_start is None:
                    yawn_start = time.time()

                mouth_elapsed = time.time() - yawn_start

            else:

                if mouth_open and yawn_start is not None:

                    duration = time.time() - yawn_start

                    if duration >= YAWN_TIME:
                        yawn_count += 1

                mouth_open = False
                yawn_start = None
                mouth_elapsed = 0

            # ==========================
            # DRIVER STATUS
            # ==========================

            status = "ALERT"
            color = (0,255,0)

            if ear < EAR_THRESHOLD:

                status = "EYES CLOSED"
                color = (0,165,255)

                if eye_elapsed >= DROWSY_TIME:

                    status = "DROWSY"
                    color = (0,0,255)

                    if not alarm_playing:

                        alarm_playing = True

                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

            if mar > MAR_THRESHOLD:

                status = "YAWNING"
                color = (255,0,255)            # ==========================
            # DROWSINESS SCORE
            # ==========================

            if status == "ALERT":
                drowsiness = 0

            elif status == "EYES CLOSED":
                drowsiness = min(
                    int((eye_elapsed / DROWSY_TIME) * 100),
                    99
                )

            elif status == "DROWSY":
                drowsiness = 100

            elif status == "YAWNING":
                drowsiness = 60

            else:
                drowsiness = 0

            current_time = datetime.now().strftime("%I:%M:%S %p")

            current = time.time()
            fps = 1 / (current - prev_time)
            prev_time = current

            alarm_status = "ON" if alarm_playing else "OFF"

            cv2.putText(
                frame,
                "DRIVER DROWSINESS DETECTION",
                (15,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

            cv2.putText(frame,f"EAR : {ear:.2f}",(15,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame,f"MAR : {mar:.2f}",(15,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame,f"Status : {status}",(15,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            cv2.putText(frame,f"Alarm : {alarm_status}",(15,160),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            cv2.putText(frame,f"Blinks : {blink_count}",(15,190),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame,f"Yawns : {yawn_count}",(15,220),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame,f"Drowsiness : {drowsiness}%",(15,250),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame,f"FPS : {int(fps)}",(15,280),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.putText(frame,current_time,(380,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )
    