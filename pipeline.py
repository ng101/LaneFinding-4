from moviepy.editor import VideoFileClip
import undistort, topview, threshold, sliding_window, draw
from find_lane import LaneFinder
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import sys

lane_finder = LaneFinder()

prefix='../CarND-Advanced-Lane-Lines/'

def debug_image(oimg):
    uimg = undistort.undistort(oimg)
    _, _, _, timg = threshold.threshold(uimg)
    wimg = topview.warp(timg)
    #result = np.dstack((timg, timg, timg))
    result = np.dstack((wimg, wimg, wimg))
    left_fit, right_fit, roc, dfc, ly, lx, ry, rx = lane_finder.find(wimg)

    #ploty = np.linspace(0, wimg.shape[0]-1, wimg.shape[0]).astype(int)
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #result = np.dstack((wimg, wimg, wimg))
    result[ly, lx] = [255, 0, 0]
    result[ry, rx] = [255, 0, 0]
    #result[ploty, left_fitx.astype(int)] = [255, 255, 0]
    #result[ploty, right_fitx.astype(int)] = [255, 255, 0]
    return result

def process_image(oimg):
    uimg = undistort.undistort(oimg)
    _, _, _, timg = threshold.threshold(uimg)
    wimg = topview.warp(timg)
    left_fit, right_fit, roc, dfc, ly, lx, ry, rx = lane_finder.find(wimg)
    result = draw.draw(uimg, wimg, left_fit, right_fit, roc, dfc, 
            ly, lx, ry, rx, False)
    return result

if __name__ == '__main__':
    output = 'project_video.mp4'
    if len(sys.argv) >= 2:
        output = sys.argv[1]

    mode = 'prod'
    if len(sys.argv) >= 3:
        mode = sys.argv[2]

    clip = VideoFileClip(prefix + output)
    if mode == 'debug':
        out_clip = clip.fl_image(debug_image)
    else:
        out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output, audio=False)
