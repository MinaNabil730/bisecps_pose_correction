import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
from io import BytesIO
import pygame

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error with TTS: {e}")

def get_motivational_message():
    messages = [
        "Great job! Keep going!",
        "You're doing awesome! Push through!",
        "Excellent work! Stay strong!",
        "Fantastic effort! You're on fire!",
        "Keep it up! You're making great progress!"
    ]
    return np.random.choice(messages)

def process_camera():
    cap = cv2.VideoCapture(0)
    t = time.time()

    reps = 0
    sets = 0
    resting = False
    last_set_time = 0

    feedback_placeholder = st.empty()
    counter_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        feedback_text = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define keypoints for the biceps curl exercise for both arms
            arms = {
                'left': {
                    'shoulder': (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])),
                    'elbow': (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])),
                    'wrist': (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
                },
                'right': {
                    'shoulder': (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])),
                    'elbow': (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])),
                    'wrist': (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))
                }
            }

            # Calculate angle for the biceps curl for both arms
            angles = {arm: calculate_angle(data['shoulder'], data['elbow'], data['wrist']) for arm, data in arms.items()}

            # Set feedback conditions and colors for both arms
            for arm, angle in angles.items():
                if angle < 100:
                    feedback_text += f"<div style='background-color:#FFDDDD; padding:10px; margin:5px; border-radius:5px;'><strong>{arm.capitalize()} arm:</strong> Fully flex your arm</div>"
                    speak("Relax")
                elif angle > 160:
                    feedback_text += f"<div style='background-color:#DDFFDD; padding:10px; margin:5px; border-radius:5px;'><strong>{arm.capitalize()} arm:</strong> Fully relax your arm</div>"
                    speak("Flex")

            # Handle rep and set counting
            left_angle = angles['left']
            right_angle = angles['right']
            angle_threshold = 160  # Example threshold for determining complete rep

            if left_angle > angle_threshold or right_angle > angle_threshold:
                if not resting:
                    reps += 1
                    resting = True
            else:
                resting = False

            # Increase sets after 10 reps and reset reps counter
            if reps >= 10:
                reps = 0
                sets += 1
                if time.time() - last_set_time > 2:  # Add a delay before speaking
                    speak(f"Great job! You have completed {sets} sets.")
                    speak(get_motivational_message())
                    last_set_time = time.time()

            # Update feedback and counter placeholders
            feedback_placeholder.markdown(f"<div style='font-size:20px; color:#333;'>{feedback_text}</div>", unsafe_allow_html=True)
            counter_placeholder.markdown(f"<div style='background-color:#E0E0E0; padding:10px; margin:5px; border-radius:5px;'><strong>Reps:</strong> {reps}<br><strong>Sets:</strong> {sets}</div>", unsafe_allow_html=True)

    cap.release()

# Streamlit UI
st.title('Biceps Curl Posture Correction')

# Display the GIF in the sidebar (it will stay fixed in the sidebar)
gif_url = "Hantelcurl.gif"  # Replace with the path or URL to your GIF
st.sidebar.image(gif_url, use_column_width=True)

# Process camera input
process_camera()
