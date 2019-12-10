from imageai.Detection import VideoObjectDetection
import os

class itemVideoSaveFile():
    def __init__(self,setModelPath):
        self.execution_path = os.getcwd()
        self.detetor = VideoObjectDetection()
        self.detetor.setModelTypeAsRetinaNet()
        self.detetor.setModelPath(os.path.join(self.execution_path,setModelPath))
        self.detetor.loadModel()
    
    def items_VideoSaveFile(self, inputFile, outputFile):
        video_path = self.detetor.detectCustomObjectsFromVideo(input_file_path=os.path.join(self.execution_path,inputFile),
        output_file_path=os.path.join(self.execution_path, outputFile),
        frames_per_second=20, log_progress=True)
        print(video_path)