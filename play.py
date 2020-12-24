import numpy as np
import cv2
import time
import math

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Titikong Drive 12 Dec 2020 (4K Footage 1).mp4")

def goodFeatures(frame):

	feats = cv2.goodFeaturesToTrack(np.mean(frame, axis =2).astype(np.uint8), 3000, qualityLevel=0.2, minDistance=4)
	kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]

	#frame3 = doORB(frame)

	frame2 = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)

	return frame2


def laneIsolation(frame):

	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	blur = cv2.GaussianBlur(gray,(5,5),0)

	#canny = cv2.Canny(blur,120,290)

	canny = cv2.Canny(blur, 50, 150)

	return canny

def region(image):
    height, width = image.shape
    triangle = np.array([[(275,height-210), (580,290), (width-200, height-225)]])
    '''triangle = np.array([
                       [(100, height), (300, 245), (width-25, height)]
                       ])'''

   	#triangle = np.array([[(100, 100), (300, 245), (width-25, height)]])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def make_points(image, average):

	sumA = np.sum(average)
	arrayNan = np.isnan(sumA)

	print("this is arrayNan: " + str(arrayNan))

	if arrayNan == False:
		
		#print(average)
		slope, y_int = average
		y1 = image.shape[0]
		#how long we want our lines to be --> 3/5 the size of the image
		y2 = int(y1 * (3/5))
		#determine algebraically
		x1 = int((y1 - y_int) // slope)
		x2 = int((y2 - y_int) // slope)
		return np.array([x1, y1, x2, y2])

def averageLine(frame, lines):

	left = []
	right = []
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)

			parameters = np.polyfit((x1,x2),(y1,y2),1)
			slope = parameters[0]
			y_int = parameters[1]
			if slope<0:
				left.append((slope, y_int))

			else:
				right.append((slope, y_int))

		#print(right)
		right_avg = np.nanmean(right, axis=0)
		left_avg = np.nanmean(left, axis=0)

		#print("this is average: " + str(left_avg))

		#right_avg = np.average(right, axis=0)
		#left_avg= np.average(left, axis=0)

		#if math.isnan(right_avg) and math.isnan(left_avg) is not True:
		#print(left_avg)

		left_line = make_points(frame, left_avg)
		right_line = make_points(frame, right_avg)
		return np.array([left_line, right_line])

def displayLines(frame, lines):

	lines_frame = np.zeros_like(frame)
	if lines is not None:
		for line in lines:
			if line is not None:

				x1,y1,x2,y2 = line

				try:
					cv2.line(lines_frame, (x1,y1), (x2,y2), (0,255,0), 14)

				except:
					print("Overflow")
				#print("this is line: " + str(line))
				#print("This is lines frame: " + str(lines_frame))
	return lines_frame

def IsolateColor(frame):

	hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

	lightWhite = np.array([0,0,100])
	darkWhite = np.array([0,0,63])

	mask = cv2.inRange(hls, lightWhite, darkWhite)

	res = cv2.bitwise_and(frame, frame, mask = mask)

	return mask

while(True):
	ret, frame = cap.read()

	frameFeat = goodFeatures(frame)
	hls = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

	red = gray[:, :, 2]

	#mask = cv2.inRange(hls, lightWhite, darkWhite)

	#res = cv2.bitwise_and(frame, frame, mask = mask)

	#frame2 = laneIsolation(hls)

	#frame3 = region(frame2)

	#frame4 = IsolateColor(frame)


	#lines = cv2.HoughLinesP(frame3, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

	#averagedLines = averageLine(frame, lines)

	#print(averagedLines)
	#blackLines = displayLines(frame, averagedLines)

	#lanes = cv2.addWeighted(frame, 0.8, blackLines, 1, 1)

	cv2.imshow('frame', red)


	#cv2.imshow('frame', frame2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
