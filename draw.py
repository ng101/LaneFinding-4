import cv2
import numpy as np
import topview

def draw(img, binary_warped, left_fit, right_fit, roc, dist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = topview.unwarp(color_warp)
    # Combine the result with the original image
    new_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(new_img, 'Radius of curvature: {:.2f}m'.format(roc), (10,100), font, 1, (255,255,255), 2)
    vehicle_placement = 'left of center' if dist > 0 else 'right of center'
    text = ('Vehicle is {:.2f}m ' + vehicle_placement).format(np.absolute(dist))
    cv2.putText(new_img, text, (10, 150), font, 1, (255,255,255), 2)
    return new_img
