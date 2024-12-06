import numpy as np

def calculate_pitch_yaw_roll(landmarks, image_width, image_height):
    def denormalize(landmark):
        return int(landmark.x * image_width), int(landmark.y * image_height), landmark.z
    
    nose_tip = denormalize(landmarks[1])
    chin = denormalize(landmarks[152])
    left_eye_corner = denormalize(landmarks[33])
    right_eye_corner = denormalize(landmarks[263])
    
    nose_to_chin = np.array(chin) - np.array(nose_tip)
    left_to_right_eye = np.array(right_eye_corner) - np.array(left_eye_corner)

    pitch = np.arctan2(nose_to_chin[2], np.linalg.norm(nose_to_chin[:2])) * (180 / np.pi)
    yaw = np.arctan2(left_to_right_eye[0], left_to_right_eye[2]) * (180 / np.pi)
    roll = np.arctan2(left_to_right_eye[1], left_to_right_eye[0]) * (180 / np.pi)
    
    return pitch, yaw, roll

def calculate_mouth_open_ratio(landmarks, image_width, image_height):
    def denormalize(landmark):
        return np.array([landmark.x * image_width, landmark.y * image_height, landmark.z])
    
    upper_lip = denormalize(landmarks[13])
    lower_lip = denormalize(landmarks[14])
    nose_tip = denormalize(landmarks[1])
    chin = denormalize(landmarks[152])
    
    mouth_open_distance = np.linalg.norm(upper_lip - lower_lip)
    face_length = np.linalg.norm(nose_tip - chin)
    
    return mouth_open_distance / face_length

def is_eye_open(landmarks, eye_landmarks):
    def distance(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z]))
    
    vertical = distance(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[1]])
    horizontal = distance(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[3]])
    
    eye_open_ratio = vertical / horizontal
    return eye_open_ratio > 0.2 