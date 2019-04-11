import face_alignment
from skimage import io
import cv2
from skimage import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from faced import FaceDetector
from faced.utils import annotate_image
import time

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd')
video = cv2.VideoCapture(0)

def draw(fr,Z):
	for i in Z:
		cv2.circle(fr,i, 2, (225,255,255), -1)
	return fr


frame_count = 0
tt_opencvHaar = 0

while True: 
	_, fr = video.read()
	predss = fa.get_landmarks(fr)
	if predss is not None:
		for preds in predss:
			Z = zip(preds[0:68,0], preds[0:68,1])
			fr = draw(fr,Z)
	##GET fps
	frame_count += 1
	t = time.time()
	tt_opencvHaar += time.time() - t
	fpsOpencvHaar = frame_count / tt_opencvHaar
	label = "FPS : {:.2f}".format(fpsOpencvHaar)
	cv2.putText(fr, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
	if frame_count == 1:
		tt_opencvHaar = 0
	#---------------------------#
	cv2.imshow('image',fr)
	if cv2.waitKey(1) == 27:
		break


cv2.destroyAllWindows()
