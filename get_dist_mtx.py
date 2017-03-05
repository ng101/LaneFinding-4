import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import glob

images = glob.glob("../CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg")

# Read in an image
img = cv2.imread('test_image.png')

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

shape = ()

for i in images:
    img = mpimg.imread(i)
    shape = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
        new_img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        name = i.split('/')[-1]
        mpimg.imsave('output_images/corners_' + name, new_img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
        shape, None, None)

with open('distortion.p', 'wb') as f:
    pickle.dump({'mtx': mtx, 'dist':dist}, f)


