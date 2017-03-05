import undistort
import pickle
import cv2
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

img = mpimg.imread('../CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')
uimg = undistort.undistort(img)

src = np.float32(
        [[600, 448], #top left
         [680, 448], #top right
         [1100, 720], #bottom right
         [200, 720]]) #bottom left

dst = np.float32(
        [[200, 0],
         [1100, 0],
         [1100, 720],
         [200, 720]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(uimg, M, (uimg.shape[1], uimg.shape[0]), flags=cv2.INTER_LINEAR)

codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
print(np.vstack([src, src[0, :]]))
src_path = Path(np.vstack([src, src[0]]).tolist(), codes)
dst_path = Path(np.vstack([dst, dst[0]]).tolist(), codes)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(uimg)
patch = patches.PathPatch(src_path, alpha=0.5, facecolor='red', lw=2)
ax1.add_patch(patch)
ax1.set_title('Original Image', fontsize=50)

ax2.imshow(warped)
patch = patches.PathPatch(dst_path, alpha=0.5, facecolor='red', lw=2)
ax2.add_patch(patch)

ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.savefig('output_images/' + 'warped-top-view.png')
plt.close(f)

with open('transform.p', 'wb') as f:
    pickle.dump({'m':M, 'minv':Minv}, f)
