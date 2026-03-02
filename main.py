import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Drawing tools
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the webcam (0 is usually the built-in laptop camera)
cap = cv2.VideoCapture(0)

# Set up the Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1, # We only need to track one driver
    refine_landmarks=True, # Gives us better eye tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Starting camera... Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to grab frame.")
            continue

        # MediaPipe needs RGB, but OpenCV uses BGR. We have to convert it.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find the face
        results = face_mesh.process(image_rgb)

        # Draw the web (tesselation) over the face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Show the video feed on your screen
        cv2.imshow('Driver Drowsiness Detector', image)

        # Listen for the 'q' key to stop the program
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up and close windows when done
cap.release()
cv2.destroyAllWindows()