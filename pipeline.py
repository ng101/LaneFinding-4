from moviepy.editor import VideoFileClip
import undistort, topview, threshold, sliding_window
import matplotlib.pyplot as plt
import cv2 
import numpy as np

prefix='../CarND-Advanced-Lane-Lines/'

def process_image(oimg):
    uimg = undistort.undistort(oimg)
    timg = threshold.threshold(uimg)
    wimg = topview.warp(timg)
    left_lane_inds, right_lane_inds, left_fit, right_fit = sliding_window.sliding_window(wimg)
    left_curverad, right_curverad = sliding_window.roc(wimg.shape[0] - 1, left_fit, right_fit)
    dfc = sliding_window.dist_from_center(wimg, left_fit, right_fit)
    roc = (left_curverad + right_curverad) / 2
    result = sliding_window.draw(uimg, wimg, left_fit, right_fit, roc, dfc)
    return result

output1 = 'project_video.mp4'
clip1 = VideoFileClip(prefix + output1)
out_clip1 = clip1.fl_image(process_image)
out_clip1.write_videofile(output1, audio=False)
