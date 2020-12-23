import numpy as np
import cv2
import time

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Titikong Drive 12 Dec 2020 (4K Footage 1).mp4')

def doCanny(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	blur = cv2.GaussianBlur(gray, (5,5), 0)

	canny = cv2.Canny(blur, 50, 150)

	return canny

def goodFeatures(frame):

	feats = cv2.goodFeaturesToTrack(np.mean(frame, axis =2).astype(np.uint8), 3000, qualityLevel=0.2, minDistance=4)
	kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]

	#frame3 = doORB(frame)

	frame2 = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)

	return frame2

def laneIsolation(frame):

	hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

	return hls


while(True):
	ret, frame = cap.read()

	#canny = doCanny(frame)
	frame3 = goodFeatures(frame)
	frame2 = laneIsolation(frame)

	frameOut = cv2.addWeighted(frame,0.6, frame3,0.7,0.7)
	cv2.imshow('frame', frameOut)
	#cv2.imshow('frame', frame2)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
