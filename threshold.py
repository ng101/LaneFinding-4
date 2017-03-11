import cv2
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import numpy as np

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
    binary_output[(grad > thresh_min) & (grad < thresh_max)] = 1
    return binary_output

def mag_grad_threshold(img, thresh_min=0, thresh_max=255, kernel=3):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    gray = img
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    grad = np.uint8(255*grad/np.max(grad))
    binary_output = np.zeros_like(grad)
    binary_output[(grad > thresh_min) & (grad < thresh_max)] = 1
    return binary_output

def dir_grad_threshold(img, thresh_min=0, thresh_max=np.pi/2, kernel=3):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    gray = img
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
    dire = np.arctan2(sobel_y, sobel_x)
    binary_output = np.zeros_like(dire).astype(np.int32)
    binary_output[(dire > thresh_min) & (dire < thresh_max)] = 1
    return binary_output

def s_threshold(img, thresh_min=0, thresh_max=255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls_img[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh_min) & (S <= thresh_max)] = 1
    return binary_output

def r_threshold(img, thresh_min=0, thresh_max=255):
    R = img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh_min) & (R <= thresh_max)] = 1
    return binary_output

def grad_threshold(img):
    abs_binary = abs_grad_threshold(img, 'x', 60, 120, 3) # abs gradient
    m_binary = mag_grad_threshold(img, 70, 140, 3)  # mag gradient
    d_binary = dir_grad_threshold(img, 0.7, 1.3, 15) # Dir gradient
    combined_binary = np.zeros_like(abs_binary)
    combined_binary[((abs_binary == 1) & (m_binary == 1)) & (d_binary == 1)] = 1
    return combined_binary

def threshold(oimg):
    r_binary = r_threshold(oimg, 225, 255) # R channel
    s_binary = s_threshold(oimg, 100, 255) # S channel
    #abs_binary = abs_grad_threshold(img, 'x', 20, 150) # abs gradient
    #m_binary = mag_grad_threshold(img, 30, 150)  # mag gradient
    #d_binary = dir_grad_threshold(img, 0.7, 1.3, 15) # Dir gradient
    combined_binary = np.zeros_like(r_binary)
    combined_binary[(r_binary == 1) | (s_binary == 1)] = 20
    combined = 255 * grad_threshold(combined_binary)
    #return r_binary, s_binary, r_thresh, s_thresh, combined
    return combined

if __name__ == '__main__':
    images = glob.glob('../CarND-Advanced-Lane-Lines/test_images/test*.jpg')
    for i in images:
        img = mpimg.imread(i)
        f,((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        
        r, s, rt, st, timg = threshold(img)

        ax2.imshow(timg, cmap='gray')
        ax2.set_title('Threshold Image', fontsize=50)

        ax3.imshow(r, cmap='gray')
        ax3.set_title('R Image', fontsize=50)

        ax4.imshow(s, cmap='gray')
        ax4.set_title('S Image', fontsize=50)

        ax5.imshow(rt, cmap='gray')
        ax5.set_title('R Grad Image', fontsize=50)

        ax6.imshow(st, cmap='gray')
        ax6.set_title('S Grad Image', fontsize=50)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        f.savefig('output_images/threshold_' + i.split('/')[-1])
        plt.close(f)
