from moviepy.editor import VideoFileClip
import undistort, topview, threshold
import matplotlib.pyplot as plt
import cv2 
import numpy as np

prefix='../CarND-Advanced-Lane-Lines/'

def process_image(img):
    img = undistort.undistort(img)
    img = threshold.threshold(img)
    img = np.dstack((np.zeros_like(img), np.zeros_like(img), 255*img))
    img = topview.warp(img)
    return img

output1 = 'project_video.mp4'
clip1 = VideoFileClip(prefix + output1)
out_clip1 = clip1.fl_image(process_image)
out_clip1.write_videofile(output1, audio=False)
