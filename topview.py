import pickle
import cv2
import glob
import matplotlib.pyplot as plt

with open('transform.p', 'rb') as f:
    data = pickle.load(f)
    M = data['m']
    Minv = data['minv']

def warp(img):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.
        INTER_LINEAR)

def unwarp(img):
    return cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.
        INTER_LINEAR)

if __name__ == '__main__':
    images = glob.glob('../CarND-Advanced-Lane-Lines/test_images/test*.jpg')
    for i in images:
        img = cv2.imread(i)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=50)
        
        wimg = warp(img)
        ax2.imshow(cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB))
        ax2.set_title('warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        f.savefig('output_images/warp_' + i.split('/')[-1])
        plt.close(f)
