from imageai.Detection import VideoObjectDetection
import os
import cv2

class itemCamSaveFile():
    def __init__(self, setModePath):
        self.execution_path = os.getcwd()
        self.detector = VideoObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(os.path.join(self.execution_path,setModePath))
        self.detector.loadModel()
    
    def item_CamSaveFile(self, camera, outputFile):
        video_path = self.detector.detectObjectsFromVideo(camera_input=camera,
            output_file_path=os.path.join(self.execution_path,outputFile),
            frames_per_second=10, log_progress=True,minimum_percentage_probability=20)
        print(video_path)