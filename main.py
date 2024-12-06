import cv2
import mediapipe as mp
import utils as utl
import tcpcon as tp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    image_height, image_width, _ = frame.shape
    if not ret:
        print("Failed to capture video.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pitch, yaw, roll = utl.calculate_pitch_yaw_roll(face_landmarks.landmark, image_width, image_height)
            mouth = utl.calculate_mouth_open_ratio(face_landmarks.landmark, image_width, image_height)
            eyes = utl.is_eye_open(face_landmarks.landmark,[159, 145, 33, 133])
            yaw = (90-yaw ) * 400
            pitch= (pitch - 0.06) * -300
            mouth=mouth*3
            eyes = 1-eyes
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            tp.move(roll,pitch,yaw,mouth,eyes)
            cv2.putText(frame, f'Pitch: {pitch:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'Yaw: {yaw:.2f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Roll: {roll:.2f}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f'Mouth: {mouth:.2f}', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'eyes: {eyes:.2f}', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('MediaPipe Face Mesh', frame)

    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
