import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import pandas as pd
import joblib  # Add to your imports at the top
import subprocess  # Add to your imports at the top
import argparse
import threading

# Debug model loading
print("=" * 50)
print("Current working directory:", os.getcwd())
model_path = 'posture_model.pkl'
print("Full model path:", os.path.abspath(model_path))
print("File exists (os.path):", os.path.exists(model_path))
print("File exists (open check):", end=" ")
try:
    with open(model_path, 'rb') as f:
        print("YES")
except FileNotFoundError:
    print("NO")

print("Directory contents:")
for file in os.listdir('.'):
    print(f"  - {file}")
print("=" * 50)

# Now try loading with more error details
try:
    model = joblib.load(model_path)
    print("ML model loaded successfully")
except Exception as e:
    print(f"Detailed error loading model: {type(e).__name__} - {str(e)}")
    # Don't exit yet, let's see what's happening
    exit(1)

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Add data collection variables
collecting_data = True  # Set to True to collect training data
posture_data = []
data_collection_count = 0
save_interval = 20  # Save to CSV every 20 samples
last_sample_time = 0
sample_cooldown = 0.1  # Half-second cooldown between samples

# Rest of your existing functions (calculate_angle, draw_angle, distance)
def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def draw_angle(frame, p1, p2, p3, angle, color):
    """Draw the angle between three points on the frame"""
    # Draw the angle lines
    cv2.line(frame, p1, p2, color, 2)
    cv2.line(frame, p2, p3, color, 2)
    
    # Put the angle text
    cv2.putText(frame, f"{angle:.1f}", 
                (p2[0] - 50, p2[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)

def distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to save collected data
def save_data():
    global posture_data, data_collection_count
    if posture_data:
        df = pd.DataFrame(posture_data)
        filename = 'posture_training_data.csv'
        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
        print(f"Saved {len(posture_data)} samples. Total: {data_collection_count}")
        posture_data = []  # Clear after saving

# Variables for calibration and alerts
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_ear_shoulder_distances = []
calibration_shoulder_widths = []
calibration_shoulder_heights = []
calibration_ear_heights = []
last_alert_time = time.time()
alert_cooldown = 3  # seconds
sound_file = "alert.mp3"  # You'll need to add a sound file or change this path

# For collecting current frame data
current_frame_data = None

# Add to calibration variables
calibration_face_widths = []
calibration_face_shoulder_ratios = []

# Add these variables with the other calibration variables
poor_posture_start_time = None

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('reminder_time', type=int, nargs='?', default=30,
                   help='Time in seconds before posture reminder')
args = parser.parse_args()

# Use the argument
poor_posture_threshold = args.reminder_time

# Add this function after your other function definitions
def notify(title, text):
    CMD = '''
    on run argv
      display notification (item 2 of argv) with title (item 1 of argv)
    end run
    '''
    subprocess.call(['osascript', '-e', CMD, title, text])

def check_recalibration():
    global is_calibrated, calibration_frames
    while True:
        if os.path.exists("recalibrate.trigger"):
            os.remove("recalibrate.trigger")
            is_calibrated = False
            calibration_frames = 0
            print("Recalibration triggered")
        time.sleep(1.0)

# Start recalibration check thread
recalibration_thread = threading.Thread(target=check_recalibration, daemon=True)
recalibration_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]))

        # Calculate basic measurements first
        shoulder_width = distance(left_shoulder, right_shoulder)
        face_width = distance(left_ear, right_ear)
        face_shoulder_ratio = face_width / shoulder_width if shoulder_width > 0 else 0

        # Calculate angles and metrics
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
        
        # Calculate additional posture metrics
        ear_to_shoulder_distance_left = distance(left_ear, left_shoulder)
        ear_to_shoulder_distance_right = distance(right_ear, right_shoulder)
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        ear_height_diff = abs(left_ear[1] - right_ear[1])

        # Store current frame data for potential sample collection
        current_frame_data = {
            'shoulder_angle': shoulder_angle,
            'neck_angle': neck_angle,
            'ear_shoulder_distance': (ear_to_shoulder_distance_left + ear_to_shoulder_distance_right) / 2,
            'shoulder_width': shoulder_width,
            'shoulder_height_diff': shoulder_height_diff,
            'ear_height_diff': ear_height_diff,
            'face_width': face_width,
            'face_shoulder_ratio': face_shoulder_ratio
        }

        # Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_ear_shoulder_distances.append((ear_to_shoulder_distance_left + ear_to_shoulder_distance_right) / 2)
            calibration_shoulder_widths.append(shoulder_width)
            calibration_shoulder_heights.append(shoulder_height_diff)
            calibration_ear_heights.append(ear_height_diff)
            calibration_face_widths.append(face_width)
            calibration_face_shoulder_ratios.append(face_shoulder_ratio)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
            neck_threshold = np.mean(calibration_neck_angles) - 10
            ear_shoulder_threshold = np.mean(calibration_ear_shoulder_distances) * 0.9
            shoulder_width_threshold = np.mean(calibration_shoulder_widths) * 0.9
            shoulder_height_threshold = np.mean(calibration_shoulder_heights) * 1.5
            ear_height_threshold = np.mean(calibration_ear_heights) * 1.5
            face_width_threshold = np.mean(calibration_face_widths) * 1.2
            face_shoulder_ratio_threshold = np.mean(calibration_face_shoulder_ratios) * 1.15
            is_calibrated = True
            print(f"Calibration complete with all metrics")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

        # Enhanced Feedback
        if is_calibrated:
            current_time = time.time()
            posture_issues = []
            
            # Use ML model for prediction
            features = np.array([[
                shoulder_angle,
                neck_angle,
                (ear_to_shoulder_distance_left + ear_to_shoulder_distance_right) / 2,
                shoulder_width,
                shoulder_height_diff,
                ear_height_diff,
                face_width,
                face_shoulder_ratio
            ]])
            prediction = model.predict(features)[0]
            
            if prediction == 'bad':
                status = "Poor Posture (ML)"
                color = (0, 0, 255)  # Red
                posture_issues.append("Sit Up Straight")
            else:
                status = "Good Posture (ML)"
                color = (0, 255, 0)  # Green

            # Determine posture status based on issues
            if posture_issues:
                status = f"Poor Posture: {', '.join(posture_issues)}"
                color = (0, 0, 255)  # Red
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green
                
            # Display data collection status on screen
            if collecting_data:
                cv2.putText(frame, f"Press 'g' for good posture, 'b' for bad posture", 
                          (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Collected: {data_collection_count} samples", 
                          (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                
            # Display the normal feedback
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Detection Method: ML Model", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset = 90  # Increased to accommodate new line
            for metric in [
                f"Shoulder Angle: {shoulder_angle:.1f}",
                f"Neck Angle: {neck_angle:.1f}",
                f"Head Forward: {((ear_to_shoulder_distance_left + ear_to_shoulder_distance_right) / 2):.1f}",
                f"Shoulder Width: {shoulder_width:.1f}",
                f"Screen Distance: {100*face_shoulder_ratio:.1f}%"
            ]:
                cv2.putText(frame, metric, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_offset += 30
            
            # Alert logic
            if prediction == 'bad':
                if poor_posture_start_time is None:
                    poor_posture_start_time = current_time
                elif current_time - poor_posture_start_time >= poor_posture_threshold:
                    if current_time - last_alert_time > alert_cooldown:
                        notify("Poor Posture Alert", ", ".join(posture_issues))
                        last_alert_time = current_time
            else:
                poor_posture_start_time = None  # Reset timer when posture is good

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    # Handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()
    
    # Sample collection with cooldown
    if collecting_data and current_frame_data and is_calibrated:
        if (key == ord('g') or key == ord('b')) and current_time - last_sample_time > sample_cooldown:
            # Create a copy with the label
            sample = current_frame_data.copy()
            sample['posture_label'] = 'good' if key == ord('g') else 'bad'
            
            # Add to dataset
            posture_data.append(sample)
            data_collection_count += 1
            
            # Update last sample time for cooldown
            last_sample_time = current_time
            
            # Show feedback
            label_type = "GOOD" if key == ord('g') else "BAD"
            print(f"Sample #{data_collection_count} recorded as {label_type} posture")
            
            # Save periodically
            if len(posture_data) >= save_interval:
                save_data()
    
    # Other controls
    if key == ord('q'):
        # Save any remaining data before quitting
        save_data()
        break
    elif key == ord('p'):
        # Toggle data collection pause/resume
        collecting_data = not collecting_data
        print(f"Data collection {'resumed' if collecting_data else 'paused'}")
    elif key == ord('d'):
        # Dump data to file immediately
        save_data()

cap.release()
cv2.destroyAllWindows()