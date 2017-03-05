import cv2
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import numpy as np

def abs_grad_threshold(img, orient='x', thresh_min=0, thresh_max=255, kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    grad = np.uint8(255*grad/np.max(grad))
    binary_output = np.zeros_like(grad)
    binary_output[(grad > thresh_min) & (grad < thresh_max)] = 1
    return binary_output

def dir_grad_threshold(img, thresh_min=0, thresh_max=np.pi/2, kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is read via mpimg
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
    dire = np.arctan2(sobel_y, sobel_x)
    binary_output = np.zeros_like(dire)
    binary_output[(dire > thresh_min) & (dire < thresh_max)] = 1
    return binary_output

def s_threshold(img, thresh_min=0, thresh_max=255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls_img[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh_min) & (S <= thresh_max)] = 1
    return binary_output

def threshold(img):
    abs_binary = abs_grad_threshold(img, 'x', 20, 100)
    m_binary = mag_grad_threshold(img, 30, 100)
    s_binary = s_threshold(img, 170, 255)
    d_binary = dir_grad_threshold(img, 0.7, 1.3, 15)
    combined_binary = np.zeros_like(abs_binary)
    combined_binary[(((abs_binary == 1) & (m_binary == 1)) | (s_binary == 1)) & (d_binary == 1)] = 1
    return combined_binary

if __name__ == '__main__':
    images = glob.glob('../CarND-Advanced-Lane-Lines/test_images/test*.jpg')
    for i in images:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        
        timg = threshold(img)
        ax2.imshow(timg, cmap='gray')
        ax2.set_title('Threshold Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        f.savefig('output_images/threshold_' + i.split('/')[-1])
        plt.close(f)
