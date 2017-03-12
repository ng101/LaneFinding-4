import cv2
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import numpy as np
import topview

def abs_grad_threshold(img, orient='x', thresh_min=0, thresh_max=255, kernel=3):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    gray = img
    if 'x' == orient:
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
       
    grad = np.absolute(grad)
    grad = np.uint8(255*grad/np.max(grad))
    binary_output = np.zeros_like(grad)
    binary_output[(grad >= thresh_min) & (grad <= thresh_max)] = 1
    return binary_output

def mag_grad_threshold(img, thresh_min=0, thresh_max=255, kernel=3):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    gray = img
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    grad = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    grad = np.uint8(255*grad/np.max(grad))
    binary_output = np.zeros_like(grad)
    binary_output[(grad >= thresh_min) & (grad <= thresh_max)] = 1
    return binary_output

def dir_grad_threshold(img, thresh_min=0, thresh_max=np.pi/2, kernel=3):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    gray = img
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
    dire = np.arctan2(sobel_y, sobel_x)
    binary_output = np.zeros_like(dire).astype(np.int32)
    binary_output[(dire >= thresh_min) & (dire <= thresh_max)] = 1
    return binary_output

def s_threshold(img, thresh_min=0, thresh_max=255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls_img[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh_min) & (S <= thresh_max)] = 1
    return binary_output

def l_threshold(img, thresh_min=0, thresh_max=255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls_img[:,:,1]
    binary_output = np.zeros_like(L)
    binary_output[(L > thresh_min) & (L <= thresh_max)] = 1
    return binary_output

def r_threshold(img, thresh_min=0, thresh_max=255):
    R = img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh_min) & (R <= thresh_max)] = 1
    return binary_output

def grad_threshold(img):
    abs_binary = abs_grad_threshold(img, 'x', 20, 100, 15) # abs x-gradient
    d_binary = dir_grad_threshold(img, 0.7, 1.3, 15) # Dir gradient
    combined_binary = np.zeros_like(abs_binary)
    cond = ((d_binary == 1) & (abs_binary == 1))
    combined_binary[cond] = 1
    return combined_binary

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    ignore_mask_color = 255    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def threshold(oimg):
    r_binary = r_threshold(oimg, 225, 255) # R channel
    s_binary = s_threshold(oimg, 80, 255) # S channel
    l_binary = l_threshold(oimg, 200, 255) # S channel
    combined_binary = (r_binary | s_binary | l_binary)
    shape = combined_binary.shape
    h, w = shape[0], shape[1]
    vertices = np.array([[(150,h), (570, 450), (715, 450), (1150,h)]], dtype=np.int32)
    combined_binary = region_of_interest(combined_binary, vertices)
    timg = 255 * grad_threshold(combined_binary)
    return r_binary, s_binary, combined_binary, timg

if __name__ == '__main__':
    images = glob.glob('../CarND-Advanced-Lane-Lines/test_images/test*.jpg')
    for i in images:
        img = mpimg.imread(i)
        f,((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        
        r, s, c, timg = threshold(img)

        ax2.imshow(timg, cmap='gray')
        ax2.set_title('Threshold Image', fontsize=50)

        ax3.imshow(c, cmap='gray')
        ax3.set_title('Com Image', fontsize=50)

        ax4.imshow(topview.warp(timg), cmap='gray')
        ax4.set_title('Warped Image', fontsize=50)

        #ax5.imshow(rt, cmap='gray')
        #ax5.set_title('R Grad Image', fontsize=50)

        #ax6.imshow(st, cmap='gray')
        #ax6.set_title('S Grad Image', fontsize=50)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        f.savefig('output_images/threshold_' + i.split('/')[-1])
        plt.close(f)
