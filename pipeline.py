from moviepy.editor import VideoFileClip
import undistort, topview, threshold, sliding_window, draw
from find_lane import LaneFinder
import matplotlib.pyplot as plt
import cv2 
import numpy as np

lane_finder = LaneFinder()

prefix='../CarND-Advanced-Lane-Lines/'

def process_image(oimg):
    uimg = undistort.undistort(oimg)
    timg = threshold.threshold(uimg)
    wimg = topview.warp(timg)
    left_fit, right_fit, roc, dfc = lane_finder.find(wimg)
    result = draw.draw(uimg, wimg, left_fit, right_fit, roc, dfc)
    return result

output1 = 'challenge_video.mp4'
clip1 = VideoFileClip(prefix + output1)
out_clip1 = clip1.fl_image(process_image)
out_clip1.write_videofile(output1, audio=False)
