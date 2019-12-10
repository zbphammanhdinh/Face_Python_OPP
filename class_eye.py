import cv2

class eye():
    def __init__(self, eye_cascade):
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade)
    
    def eye_draw(self, frame, gray):
        eye_rects = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=11)
        for (x,y,w,h) in eye_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame, "eye", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
    
    def eye_draw1(self, frame, gray, eyes):
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, "eye", (x, y),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)