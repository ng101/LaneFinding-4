import cv2
import numpy as np
import sliding_window

class LaneFinder():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def find(self, bwimg):
        leftx, lefty, rightx, righty = sliding_window.sliding_window(bwimg)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_roc, right_roc = sliding_window.roc(bwimg.shape[0] - 1, left_fit, right_fit)
        dfc = sliding_window.dist_from_center(bwimg, left_fit, right_fit)
        roc = (left_roc + right_roc) / 2
        return left_fit, right_fit, roc, dfc
