from threading import Thread
import face_recognition
from cv2 import cv2
from scipy.spatial import distance as dis 
import winsound
import numpy as np

MIN_EYE_ASPECT_RATIO = 0.30
EYE_ASPECT_CONSECUTIVE_FRAMES = 10
COUNTER = 0
ALARM_ON = False

def sound_alarm(soundFile):
    # Play the sound file
    winsound.PlaySound(soundFile, winsound.SND_FILENAME)

def eye_aspect_ratio(eye):
    A = dis.euclidean(eye[1], eye[5])
    B = dis.euclidean(eye[2], eye[4])
    C = dis.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 400)
    video_capture.set(4, 400)
    while True:
        ret, frame = video_capture.read()
        ear = 0  # Initialize ear variable outside the loop
        face_landmark_list = face_recognition.face_landmarks(frame)
        for face_landmark in face_landmark_list:
            leftEye = face_landmark['left_eye']
            rightEye = face_landmark['right_eye']
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2
            
            left_pts = np.array(leftEye)
            right_pts = np.array(rightEye)
            
            cv2.polylines(frame, [left_pts], True, (255, 0, 0), 1)
            cv2.polylines(frame, [right_pts], True, (255, 0, 0), 1)
            
            if ear < MIN_EYE_ASPECT_RATIO:
                COUNTER += 1
                if COUNTER >= EYE_ASPECT_CONSECUTIVE_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm, args=('alarm.wav',))
                        t.daemon = True
                        t.start()
                        print("Starting alarm thread...")
                cv2.putText(frame, "Falling Asleep", (5, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
            else:
                COUNTER = 0
                ALARM_ON = False
        cv2.putText(frame, "Ear: {:.2f}".format(ear), (300, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "Counter: {}".format(COUNTER), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("sleep detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
