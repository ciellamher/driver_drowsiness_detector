import cv2
import mediapipe as mp
import math
import pygame 
import numpy as np
import time
from threading import Thread

# --- MULTI-THREADED CAMERA CLASS ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.success, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.success, self.frame = self.capture.read()
        self.capture.release()

    def read(self):
        return self.success, self.frame

    def stop(self):
        self.stopped = True
# -----------------------------------

pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('alarm.wav')
except pygame.error:
    print("WARNING: Could not find 'alarm.wav'.")
    alarm_sound = None

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
POSE_LANDMARKS = [1, 33, 263, 61, 291, 199] 

EAR_THRESHOLD = 0.25      
PITCH_THRESHOLD = -10.0   
CONSECUTIVE_FRAMES = 20   
frame_counter = 0         
missing_face_counter = 0  

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(eye_points):
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    h = euclidean_distance(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

cap = ThreadedCamera(0).start()
prev_time = 0 

with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Starting final UI build... Press 'q' to quit.")

    while True:
        success, image = cap.read()
        if not success:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        ih, iw, _ = image.shape

        # --- UI OVERLAY SETUP ---
        overlay = image.copy()
        # Draw a semi-transparent black rectangle at the top
        cv2.rectangle(overlay, (0, 0), (iw, 90), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Default status
        status_text = "STATUS: AWAKE"
        status_color = (0, 255, 0) # Green
        # ------------------------

        if results.multi_face_landmarks:
            missing_face_counter = 0 
            
            for face_landmarks in results.multi_face_landmarks:
                right_eye_coords = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in RIGHT_EYE]
                left_eye_coords = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in LEFT_EYE]

                avg_ear = (calculate_ear(right_eye_coords) + calculate_ear(left_eye_coords)) / 2.0
                
                face_2d = np.array([[int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)] for i in POSE_LANDMARKS], dtype=np.float64)
                face_3d = np.array([[int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih), face_landmarks.landmark[i].z] for i in POSE_LANDMARKS], dtype=np.float64)
                
                focal_length = 1 * iw
                cam_matrix = np.array([[focal_length, 0, iw / 2], [0, focal_length, ih / 2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch = angles[0] * 360

                # Print Core Metrics inside the overlay
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"PITCH: {pitch:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Drowsiness Logic
                if avg_ear < EAR_THRESHOLD or pitch < PITCH_THRESHOLD:
                    frame_counter += 1  
                    status_text = f"WARNING ({frame_counter}/{CONSECUTIVE_FRAMES})"
                    status_color = (0, 165, 255) # Orange

                    if frame_counter >= CONSECUTIVE_FRAMES:
                        status_text = "DROWSINESS DETECTED!"
                        status_color = (0, 0, 255) # Red
                        # Big flashing alert in the center of the screen
                        cv2.rectangle(image, (0, ih // 2 - 50), (iw, ih // 2 + 50), (0, 0, 255), -1)
                        cv2.putText(image, "WAKE UP!", (iw // 2 - 150, ih // 2 + 20), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 5)
                        
                        if alarm_sound and not pygame.mixer.get_busy(): alarm_sound.play()
                else:
                    frame_counter = 0
                    if alarm_sound and pygame.mixer.get_busy(): pygame.mixer.stop()
        else:
            missing_face_counter += 1
            status_text = f"SEARCHING FACE ({missing_face_counter}/10)"
            status_color = (0, 165, 255) # Orange
            
            if missing_face_counter >= 10:
                status_text = "NO DRIVER DETECTED!"
                status_color = (0, 0, 255) # Red
                cv2.putText(image, "DRIVER MISSING", (iw // 2 - 200, ih // 2 + 20), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)
                if alarm_sound and not pygame.mixer.get_busy(): alarm_sound.play()

        # Add FPS and Dynamic Status Text to the overlay
        cv2.putText(image, status_text, (200, 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        cv2.putText(image, f"FPS: {int(fps)}", (iw - 130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Driver Drowsiness Detector', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.stop()
cv2.destroyAllWindows()