import cv2
from class_eye import eye
from class_mouth import mouth
from class_nose import nose

class faceAuto(eye,mouth,nose):
    def __init__(self, face, eye_cas, mouth_cas, nose_cas):
        eye.__init__(self, eye_cascade = eye_cas)
        mouth.__init__(self, mouth_cascade = mouth_cas)
        nose.__init__(self, nose_cascade = nose_cas)
        self.face_cas = cv2.CascadeClassifier(face)
    
    def faceAuto_draw(self, frame, gray):
        face_rects = self.face_cas.detectMultiScale(
            gray,
            scaleFactor= 1.7,
            minNeighbors=10
        )
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, "face", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_frame = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.3,minNeighbors=5)
            noses = self.nose_cascade.detectMultiScale(roi_gray,scaleFactor=1.3,minNeighbors=5)
            mouths = self.mouth_cascade.detectMultiScale(roi_gray,scaleFactor=1.3,minNeighbors=5)

            eye.eye_draw1(self,roi_frame,roi_gray,eyes)
            mouth.mouth_draw1(self,roi_frame,roi_gray,mouths)
            nose.nose_draw1(self,roi_frame,roi_gray,noses)