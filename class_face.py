import cv2
from class_eye import eye
from class_nose import nose
from class_mouth import mouth

class face():
    def __init__(self, face_cascade):
        self.face_cascade = cv2.CascadeClassifier(face_cascade)
    
    def face_draw(self, frame, gray):
        face_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=11
            )
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame, "face", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
    