import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# These are the exact numerical indexes for the 6 points around each eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

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

        # Convert colors for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Get the height and width of your webcam feed
        ih, iw, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Loop through the RIGHT eye points and draw a green dot on each
                for index in RIGHT_EYE:
                    # MediaPipe gives coordinates as percentages, so we multiply by the image size
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1) 
                
                # Loop through the LEFT eye points and draw a green dot on each
                for index in LEFT_EYE:
                    x = int(face_landmarks.landmark[index].x * iw)
                    y = int(face_landmarks.landmark[index].y * ih)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1) 

        # Show the video feed
        cv2.imshow('Driver Drowsiness Detector', image)

        # Press 'q' to close the window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()