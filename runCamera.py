import cv2

from faced import FaceDetector
from faced.utils import annotate_image
from model import FaceKeypointsCaptureModel
import numpy as np

rgb = cv2.VideoCapture(0)
face_detector = FaceDetector()
facekeypont_detector = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
thresh = 0.7

COLUMNS = ['left_eye_center_x', 'left_eye_center_y',
               'right_eye_center_x', 'right_eye_center_y',
               'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 
               'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
               'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 
               'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
               'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
               'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
               'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
               'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
               'nose_tip_x', 'nose_tip_y',
               'mouth_left_corner_x', 'mouth_left_corner_y',
               'mouth_right_corner_x', 'mouth_right_corner_y',
               'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
               'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

while True: 

    _, fr = rgb.read()
    rgb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	# Receives RGB numpy image (HxWxC) and
	# returns (x_center, y_center, width, height, prob) tuples. 
    bboxes = face_detector.predict(rgb_img, thresh)


    for (x,y,w,h,p) in bboxes:
        x = int(x-w/2)
        y = int(y-h/2)
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (96, 96))
        pred, pred_dict = facekeypont_detector.predict_points(roi[np.newaxis, :, :, np.newaxis])
        pred, pred_dict = facekeypont_detector.scale_prediction((x, fc.shape[1]+x), (y, fc.shape[0]+y))
        listValue = []
        for col in COLUMNS:
            listValue.append(pred_dict[col])
        for i in range(0, 29, 2):
            cv2.circle(fr,(listValue[i], listValue[i+1]), 1, (0,255,0), -1)
	#Use this utils function to annotate the image.
    ann_img = annotate_image(fr, bboxes)

	# Show the image
    cv2.imshow('image',fr)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()