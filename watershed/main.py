import cv2
import numpy as np
import matplotlib.pyplot as plt
# 加载图像
img = cv2.imread('hutao.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值分割
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 先腐蚀再膨胀
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 得到背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# sure_bg与sure_fg相减,得到前景背景重合区域
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)

# 连通区域处理
ret, markers = cv2.connectedComponents(sure_fg,connectivity=8)
markers = markers + 1
markers[unknow==255] = 0   

# 分水岭算法
markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为-1

img[markers == -1] = [255, 0, 0]   # 标注为-1 的像素点标蓝
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
