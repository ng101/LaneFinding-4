import cv2
import numpy as np
import sliding_window

class LaneFinder:
    MAX_LANE_GAPS = 5 # after which to re-run sliding window
    SMOOTHING_COUNT = 5 # smooth fit over these many last fits

    def __init__(self):
        self.last_lane_counter = -1000 # last time lane was detected
        self.counter = 0 # Number of frames processed
        self.lane_width_fits = []
        self.last_lane_width_fit = None
        self.lane_width = 0
        self.reset()

    def reset(self):
        self.left_fits = []
        self.right_fits = []
        self.last_left_fit  = None
        self.last_right_fit = None
        self.leftx = []
        self.lefty = []
        self.rightx = []
        self.righty = []

    def save(self, ly, lx, ry, rx, lfit, rfit):
        self.lefty.append(ly)
        self.leftx.append(lx)
        self.righty.append(ry)
        self.rightx.append(rx)
        self.left_fits.append(lfit)
        self.right_fits.append(rfit)
        self.last_left_fit = lfit
        self.last_right_fit = rfit
        self.lane_width_fits.append(rfit - lfit)

        # estimate new width
        widths = self.lane_width_fits[-1 * LaneFinder.SMOOTHING_COUNT:]
        self.last_lane_width_fit = np.mean(np.array(widths), axis=0)
        self.lane_width = LaneFinder.get_x(0, self.last_lane_width_fit)

    def partial_save(self, y, x, fit, side):
        if side == 'l':
            self.lefty.append(y)
            self.leftx.append(x)
            self.left_fits.append(fit)
            self.last_left_fit = fit
        else:
            self.righty.append(y)
            self.rightx.append(x)
            self.right_fits.append(fit)
            self.last_right_fit = fit

    @staticmethod
    def get_x(y, fit):
        return fit[0]*y**2 + fit[1]*y + fit[2]

    @staticmethod
    def parallel_lines(eval_y_list, lfit, rfit, min_diff = 500, max_diff = 1500):
        xl = np.array([LaneFinder.get_x(y, lfit) for y in eval_y_list])
        xr = np.array([LaneFinder.get_x(y, rfit) for y in eval_y_list])
        l = len(eval_y_list)
        diff = xr - xl
        bool_max = (np.sum(diff <= max_diff) == l)
        bool_min = (np.sum(diff >= min_diff) == l)
        return bool_max and bool_min
 
    @staticmethod
    def same_lines(eval_y_list, fita, fitb, max_diff = 200):
        xa = np.array([LaneFinder.get_x(y, fita) for y in eval_y_list])
        xb = np.array([LaneFinder.get_x(y, fitb) for y in eval_y_list])
        l = len(eval_y_list)
        diff = np.absolute(xa - xb)
        return np.sum(diff <= max_diff) == l
 
    @staticmethod
    def sane_line(y, x, last_fit = None):
        if len(x) < 100:
            #print('Failed pixel check')
            return False, np.array([0, 0, 0])

        # Fit a second order polynomial to each
        fit = np.polyfit(y, x, 2)
        if last_fit is not None:
            if not LaneFinder.same_lines([0, 719], fit, last_fit):
                #print('Diverging line')
                return False, np.array([0, 0, 0])
        return True, fit


    def smoothen(self, side):
        SMOOTHING_COUNT = LaneFinder.SMOOTHING_COUNT
        if side == 'l':
            x = np.concatenate(self.leftx[-1 * SMOOTHING_COUNT:])
            y = np.concatenate(self.lefty[-1 * SMOOTHING_COUNT:])
        else:
            x = np.concatenate(self.rightx[-1 * SMOOTHING_COUNT:])
            y = np.concatenate(self.righty[-1 * SMOOTHING_COUNT:])
        return np.polyfit(y, x, 2)

    def parallel_check(self, curr_lfit, curr_rfit):
        width = self.lane_width
        min_width = max(500, 0.75 * width)
        max_width = min(1200, 1.25 * width)
        return LaneFinder.parallel_lines([0, 719],
                curr_lfit, curr_rfit, min_width, max_width)

    def find1(self, bwimg):
        leftx, lefty, rightx, righty = sliding_window.sliding_window(bwimg)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_roc, right_roc = sliding_window.roc(bwimg.shape[0] - 1, left_fit, right_fit)
        dfc = sliding_window.dist_from_center(bwimg, left_fit, right_fit)
        roc = (left_roc + right_roc) / 2
        return left_fit, right_fit, roc, dfc

    def find(self, bwimg):
        # Set current counter
        self.counter += 1

        # Check if we need to re-run sliding window
        reset = False
        if self.counter - self.last_lane_counter >= LaneFinder.MAX_LANE_GAPS:
            reset = True

        last_l_fit = self.last_left_fit
        last_r_fit = self.last_right_fit
        if (reset is True) or (last_l_fit is None) or (last_r_fit is None):
            self.reset() # reset history, start afresh
            lx, ly, rx, ry = sliding_window.sliding_window(bwimg)
        else:
            lx, ly, rx, ry = sliding_window.targeted_lane_search(
                    bwimg, last_l_fit, last_r_fit)

        lval, lfit = LaneFinder.sane_line(ly, lx, last_l_fit)
        rval, rfit = LaneFinder.sane_line(ry, rx, last_r_fit)
        parallel = self.parallel_check(lfit, rfit)
        
        if lval and rval and parallel:
            #print('Sane Lane')
            self.last_lane_counter = self.counter
            self.save(ly, lx, ry, rx, lfit, rfit)
            lfit = self.smoothen('l')
            rfit = self.smoothen('r')
        elif lval is True and self.last_lane_width_fit is not None:
            #print('Left Line valid')
            self.partial_save(ly, lx, lfit, 'l')
            lfit = self.smoothen('l')
            rfit = lfit + self.last_lane_width_fit
        elif rval is True and self.last_lane_width_fit is not None:
            #print('Right Lane valid')
            self.partial_save(ry, rx, rfit, 'r')
            rfit = self.smoothen('r')
            lfit = rfit - self.last_lane_width_fit
        elif len(self.leftx) > 0 and len(self.rightx) > 0:
            #print('Approximating from last')
            lfit = self.smoothen('l')
            rfit = self.smoothen('r')
        #else:
            #print('No good fit found')

        left_roc, right_roc = sliding_window.roc(bwimg.shape[0] - 1, lfit, rfit)
        dfc = sliding_window.dist_from_center(bwimg, lfit, rfit)
        roc = (left_roc + right_roc) / 2
        return lfit, rfit, roc, dfc
