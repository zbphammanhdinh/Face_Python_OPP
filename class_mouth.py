import cv2

class mouth():
    def __init__(self, mouth_cascade):
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade)

    def mouth_draw(self, frame, gray):
        mouth_rects = self.mouth_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.7, 
            minNeighbors=10
            )
        for (x,y,w,h) in mouth_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame, "mouth", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
    
    def mouth_draw1(self,frame, gray, mouths):
        for (x,y,w,h) in mouths:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame, "mouth", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)