import cv2
import mediapipe as mp
import math
import pygame  # NEW: Import the audio library

# --- NEW: Initialize the audio mixer and load the sound ---
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('alarm.wav')
except pygame.error:
    print("WARNING: Could not find 'alarm.wav'. Make sure the file is in your project folder!")
    alarm_sound = None
# ----------------------------------------------------------

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25      
CONSECUTIVE_FRAMES = 20   
frame_counter = 0         

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(eye_points):
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    h = euclidean_distance(eye_points[0], eye_points[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Starting camera... Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        ih, iw, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                right_eye_coords = []
                for index in RIGHT_EYE:
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    right_eye_coords.append((x, y))
                    
                left_eye_coords = []
                for index in LEFT_EYE:
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    left_eye_coords.append((x, y))

                right_ear = calculate_ear(right_eye_coords)
                left_ear = calculate_ear(left_eye_coords)
                avg_ear = (right_ear + left_ear) / 2.0
                
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- AUDIO ALERT LOGIC ---
                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1  
                    
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        cv2.putText(image, "DROWSINESS DETECTED!", (10, 300), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        
                        # NEW: Play the alarm if it isn't already playing
                        if alarm_sound and not pygame.mixer.get_busy():
                            alarm_sound.play()
                else:
                    frame_counter = 0
                    # NEW: Stop the alarm the moment the eyes open!
                    if alarm_sound and pygame.mixer.get_busy():
                        pygame.mixer.stop()
                # -------------------------

        cv2.imshow('Driver Drowsiness Detector', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()