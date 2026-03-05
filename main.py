import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Exact indexes for the 6 points around each eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Math function to calculate the distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    # Vertical distances
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    # Horizontal distance
    h = euclidean_distance(eye_points[0], eye_points[3])
    
    # EAR Formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up the Face Mesh model
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
                
                # 1. Grab coordinates for the right eye
                right_eye_coords = []
                for index in RIGHT_EYE:
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    right_eye_coords.append((x, y))
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    
                # 2. Grab coordinates for the left eye
                left_eye_coords = []
                for index in LEFT_EYE:
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    left_eye_coords.append((x, y))
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # 3. Calculate the EAR for both eyes
                right_ear = calculate_ear(right_eye_coords)
                left_ear = calculate_ear(left_eye_coords)
                
                # 4. Find the average EAR
                avg_ear = (right_ear + left_ear) / 2.0
                
                # 5. Print the EAR value onto the video screen
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the video feed
        cv2.imshow('Driver Drowsiness Detector', image)

        # Press 'q' to close the window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()