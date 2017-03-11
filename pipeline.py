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
    _, _, _, timg = threshold.threshold(uimg)
    wimg = topview.warp(timg)
    #result = np.dstack((timg, timg, timg))
    #result = np.dstack((wimg, wimg, wimg))
    #plt.imshow(result)
    #plt.show()
    left_fit, right_fit, roc, dfc, ly, lx, ry, rx = lane_finder.find(wimg)

    #ploty = np.linspace(0, wimg.shape[0]-1, wimg.shape[0]).astype(int)
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #result = np.dstack((wimg, wimg, wimg))
    #result[ly, lx] = [255, 0, 0]
    #result[ry, rx] = [255, 0, 0]
    #result[ploty, left_fitx.astype(int)] = [255, 255, 0]
    #result[ploty, right_fitx.astype(int)] = [255, 255, 0]
    #valid_index =
    result = draw.draw(uimg, wimg, left_fit, right_fit, roc, dfc, 
            ly, lx, ry, rx, False)
    return result

output1 = 'challenge_video.mp4'
clip1 = VideoFileClip(prefix + output1)
out_clip1 = clip1.fl_image(process_image)
out_clip1.write_videofile(output1, audio=False)
