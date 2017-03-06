import cv2
import numpy as np
import sliding_window

class LaneFinder:
    def __init__(self):
        self.last_lane_counter = -1000 # last time lane was detected
        self.max_lane_gaps = 5 # after which to re-run sliding window
        self.counter = 0 # Number of frames processed
        self.left_fits = []
        self.right_fits = []

    @staticmethod
    def get_x(y, fit):
        return fit[0]*y**2 + fit[1]*y + fit[2]

    def sanity_check(self, leftx, lefty, rightx, righty, left_fit, right_fit):
        pixel_check = (len(leftx) > 200) and (len(rightx) > 200)

        # parallel check
        # Approx distance between lines should be same at top and bottom of the
        # image and should be greater than 500 px
        y_top = 0
        y_bot = 719
        top_pll = (LaneFinder.get_x(y_top, right_fit) - LaneFinder.get_x(y_top, left_fit)) > 500
        bot_pll = (LaneFinder.get_x(y_bot, right_fit) - LaneFinder.get_x(y_bot, left_fit)) > 500
        return pixel_check and top_pll and bot_pll

    def smoothen(self):
        # smoothen over last 5 detections,
        # considering only high confidence detections
        smoothing_count = 5
        lefts = self.left_fits[-1 * smoothing_count:]
        rights = self.right_fits[-1 * smoothing_count:]

        left = np.mean(np.array(lefts), axis=0)
        right = np.mean(np.array(rights), axis=0)
        return left, right

    def find(self, bwimg):
        # Set current counter
        self.counter += 1

        # Check if we need to re-run sliding window
        reset = False
        if self.counter - self.last_lane_counter >= self.max_lane_gaps:
            reset = True

        if reset is True:
            leftx, lefty, rightx, righty = sliding_window.sliding_window(bwimg)
        else:
            leftx, lefty, rightx, righty = sliding_window.targeted_lane_search(
                    bwimg, self.left_fits[-1], self.right_fits[-1])

        if reset is True:
            self.left_fits = []   # Discard earlier fits
            self.right_fits = []

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Do a sanity check on basis of which, 
        # the result may or may not be store in history
        if self.sanity_check(leftx, lefty, rightx, righty, left_fit, right_fit):
            self.last_lane_counter = self.counter
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)
            left_fit, right_fit = self.smoothen()
        elif len(self.left_fits) > 0:
            left_fit, right_fit = self.smoothen()
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)

        left_roc, right_roc = sliding_window.roc(bwimg.shape[0] - 1, left_fit, right_fit)
        dfc = sliding_window.dist_from_center(bwimg, left_fit, right_fit)
        roc = (left_roc + right_roc) / 2
        return left_fit, right_fit, roc, dfc
