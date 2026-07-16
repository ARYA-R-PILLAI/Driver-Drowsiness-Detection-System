#  AI Driver Drowsiness Detection System

An AI-powered Driver Drowsiness Detection System built using **Python, Flask, OpenCV, and MediaPipe Face Landmarker**. The system monitors a driver's face in real-time through a webcam, detects eye closure and yawning, and triggers an alarm whenever signs of drowsiness are observed.

---

##  Features

-  Real-time eye closure detection using Eye Aspect Ratio (EAR)
-  Yawn detection using Mouth Aspect Ratio (MAR)
-  Audio alarm for prolonged eye closure
-  Live dashboard displaying:
  - Driver Status
  - EAR
  - MAR
  - Blink Count
  - Yawn Count
  - Drowsiness Percentage
  - FPS
  - Current Time
-  Flask web interface
-  Face landmark detection using MediaPipe Face Landmarker

---

##  Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe Tasks
- HTML
- CSS
- NumPy
- Threading

---

## 📂 Project Structure

```
Driver-Drowsiness-Detection/
│
├── app.py                  # Flask application
├── main.py                 # AI detection logic
├── alarm.wav               # Alarm sound
├── face_landmarker.task    # MediaPipe face landmark model
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### Install dependencies

```bash
pip install flask
pip install opencv-python
pip install mediapipe
pip install playsound
```

---

##  Run the Project

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

Allow webcam access when prompted.

---

##  Detection Logic

### Eye Aspect Ratio (EAR)

The system calculates the Eye Aspect Ratio from facial landmarks.

- EAR below threshold → Eyes Closed
- Eyes closed continuously → Driver is Drowsy

---

### Mouth Aspect Ratio (MAR)

The system calculates the Mouth Aspect Ratio.

- MAR above threshold → Yawning detected

---

## 🚨 Driver Status

The application displays one of the following statuses:

- ALERT ✅
- EYES CLOSED ⚠️
- DROWSY 🚨
- YAWNING 😴
- NO FACE

---

##  User Interface

The dashboard displays:

- Live Camera Feed
- Driver Status
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Blink Counter
- Yawn Counter
- Drowsiness Percentage
- Alarm Status
- FPS
- Current Time

---

##  Future Improvements

- Head pose estimation
- Mobile application support
- Driver identity recognition
- Fatigue analytics
- Trip history database
- SMS/Email alerts
- Cloud deployment
- Night vision optimization

---

##  Author

**Arya R Pillai**

Information Science Engineering Student

---

##  License

This project is developed for educational and academic purposes.
