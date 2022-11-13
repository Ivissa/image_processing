import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def put(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, plt.cm.gray), plt.title('原图'), plt.axis('off')
    plt.savefig('1.new.jpg')
    rows, cols = img.shape[:2]

    # 图像顺时针旋转60度
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -60, 1)
    rot = cv2.warpAffine(img, M, (rows, cols))
    plt.imshow(rot, plt.cm.gray), plt.title('旋转'), plt.axis('off')
    plt.savefig('旋转')
    plt.show()
    # 图像向右平移
    H = np.float32([[1, 0, 200], [0, 1, 0]])
    tra = cv2.warpAffine(img, H, (rows, cols))
    fuhe = cv2.warpAffine(rot, H, (rows, cols))
    plt.imshow(tra, plt.cm.gray), plt.title('平移'), plt.axis('off')
    plt.savefig('平移')
    plt.show()
    # 翻转
    tra = cv2.flip(img, 1)
    fuhe = cv2.flip(fuhe, 1)
    plt.imshow(tra, plt.cm.gray), plt.title('翻转'), plt.axis('off')
    plt.savefig('翻转')
    plt.show()

    plt.imshow(fuhe, plt.cm.gray), plt.title('复合'), plt.axis('off')
    plt.savefig('复合')
    plt.show()

    height = img.shape[0]
    width = img.shape[1]

    result = np.zeros(img.shape, np.uint8)

    # 图像线性点运算 DB=DA×1.5
    for i in range(height):
        for j in range(width):

            if (int(img[i, j] * 1.5) > 255):
                gray = 255
            else:
                gray = int(img[i, j] * 1.5)

            result[i, j] = np.uint8(gray)

    # 显示图像
    plt.imshow(result, plt.cm.gray), plt.title('线性变化'), plt.axis('off')
    plt.savefig('线性')
    plt.show()


def linear_transform(path):##分段线性
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    r1, s1 = 40, 10
    r2, s2 = 160, 200
    k1 = s1 / r1  # 第一段斜率
    k2 = (s2 - s1) / (r2 - r1)  # 第二段斜率
    k3 = (255 - s2) / (255 - r2)  # 第三段斜率
    img_copy = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            if img[i, j] < r1:
                img_copy[i, j] = k1 * img[i, j]
            elif r1 <= img[i, j] <= r2:
                img_copy[i, j] = k2 * (img[i, j] - r1) + s1
            else:
                img_copy[i, j] = k3 * (img[i, j] - r2) + s2

    plt.imshow(img_copy, plt.cm.gray), plt.title('分段线性'), plt.axis('off')
    plt.savefig('分段线性')
    plt.show()


def log(path):##非线性变化
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = 1 * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    plt.imshow(output, plt.cm.gray), plt.title('非线性'), plt.axis('off')
    plt.savefig('非线性')
    plt.show()


# 调用
put(r'./Fig0413Gry.bmp')
linear_transform(r'./Fig0413Gry.bmp')
log(r'./Fig0413Gry.bmp')



