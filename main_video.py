import cv2
from class_mouth import mouth
from class_eye import eye
from class_face import face
from class_nose import nose
from class_items import items

from class_faceAuto import faceAuto

from class_itemCamSaveFile import itemCamSaveFile
from class_itemVideoSaveFile import itemVideoSaveFile


# ------------ Sử dụng camera ----------------
video_capture = cv2.VideoCapture(0)
face_cat = face(face_cascade = "haarcascade_frontalface_default.xml")
eye_cat = eye(eye_cascade = "haarcascade_eye.xml")
mouth_cat = mouth(mouth_cascade ="haarcascade_mcs_mouth.xml")
nose_cat = nose(nose_cascade = "haarcascade_mcs_nose.xml")

faceAuto_cat = faceAuto(
    face = "haarcascade_frontalface_default.xml",
    eye_cas = "haarcascade_eye.xml",
    mouth_cas = "haarcascade_mcs_mouth.xml",
    nose_cas = "haarcascade_mcs_nose.xml"
)

items_cat = items(detectorModel="hololens-ex-60--loss-2.76.h5", setJson="detection_config.json")

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Note1:
    #face_cat.face_draw(frame,gray)
    #eye_cat.eye_draw(frame,gray)
    #mouth_cat.mouth_draw(frame,gray)
    #nose_cat.nose_draw(frame,gray)

    # Note2:
    faceAuto_cat.faceAuto_draw(frame,gray)

    # Note3:
    # items_cat.items_draw(frame)

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# ------------ Sử dụng camera ===> lưu file -------------

'''camera = cv2.VideoCapture(0)
fileCam_cat = itemCamSaveFile(setModePath="yolo.h5")
fileCam_cat.item_CamSaveFile(camera=camera, outputFile="camera_detected_video")'''

# ------------ Sử dụng file Video----------------------
'''fileVideo_cat = itemVideoSaveFile(setModelPath="resnet50_coco_best_v2.0.1.h5")
fileVideo_cat.items_VideoSaveFile(inputFile="traffic.mp4", outputFile="traffic_detected")
'''