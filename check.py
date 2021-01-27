import cv2
import numpy as np
import os 
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (19,19)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Creating Vectors of the space
objpoints = []
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('check/*.JPEG')

for frame in images:
	img = cv2.imread(frame)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

	if ret == True:
		objpoints.append(objp)
		# refining pixel coordinates for given 2d points.
		corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
		imgpoints.append(corners2)
		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

	resize = cv2.resize(img, (540,720))

	cv2.imshow('img', resize)

	cv2.waitKey(0)

cv2.destroyAllWindows()