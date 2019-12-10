from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2

class items():
    def __init__(self, detectorModel, setJson):
        self.detector = CustomObjectDetection()
        self.detector.setModelPath(detectorModel)
        self.detector.setJsonPath(setJson)
        self.detector.loadModel()
    
    def items_draw(self, frame):
        detected_image, detections = self.detector.detectObjectsFromImage(input_image=frame,input_type="array",output_type="array")
        for detection in detections:
            print(detection["name"], " : ", detection["percentage_probability"])
            (x1,y1,x2,y2) = detection["box_points"]
            print("x1: ", x1," - y1: ", y1," - x2: ", x2," - y2: ", y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, detection["name"],(x1 + 6, y1 - 6), font, 1.0, (255,255,255), 1)