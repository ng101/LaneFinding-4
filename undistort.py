import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

with open('distortion.p', 'rb') as f:
    data = pickle.load(f)
    mtx = data['mtx']
    dist = data['dist']

def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

if __name__ == '__main__':
    prefix = '../CarND-Advanced-Lane-Lines/camera_cal/'
    imgname = prefix + 'calibration1.jpg'
    img = mpimg.imread(imgname)
    uimg = undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(uimg)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/' + 'undistort.png')
    plt.close(f)
