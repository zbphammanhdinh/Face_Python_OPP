import cv2

class nose():
    def __init__(self, nose_cascade):
        self.nose_cascade = cv2.CascadeClassifier(nose_cascade)

    def nose_draw(self, frame, gray):
        nose_rects = self.nose_cascade.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=11
        )
        for (x,y,w,h) in nose_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    def nose_draw1(self, frame, gray, nose):
        for (x,y,w,h) in nose:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)