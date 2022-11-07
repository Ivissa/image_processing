import cv2
import numpy as np
import io
import math
from skimage.metrics import structural_similarity

img = cv2.imread('hutao.png')
h, w = img.shape[:2]
yvu = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, v, u = cv2.split(yvu)

# 下采样 U 和 V
u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))  # 要注意 w或h 为单数的情况
v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))

f = io.BytesIO()
f.write(y.tobytes())
f.write(u.tobytes())
f.write(v.tobytes())
f.seek(0)

img_np = np.frombuffer(f.read(), np.uint8)

img_yuv_len = img_np.size
img_y_len = h * w

y = img_np[:img_y_len]
u = img_np[img_y_len:(img_yuv_len - img_y_len) // 2 + img_y_len]
v = img_np[(img_yuv_len - img_y_len) // 2 + img_y_len:]

# DPCM量化编码，采用边编码，边解码的形式
img_re = np.zeros(img_y_len, np.uint16)
yprebuff = np.zeros(img_y_len, np.uint16)
radio=512/(1<<8)  # 量化因子
for i in range(h):
    for j in range(w):
        if j == 0:
            ypre = y[j + i * w]-128  # 计算预测误差
            yprebuff[j + i * w] = (ypre+255)/radio  # 量化预测误差
            img_re[j + i * w] = (yprebuff[j + i * w]-255/radio)*radio+128  # 重建像素,j解码
            if img_re[j + i * w]>255:
                img_re[j + i * w] = 255
            yprebuff[j + i * w] = yprebuff[j + i * w]*radio/2

        else:
            ypre = y[j + i * w] - img_re[j + i * w - 1]  # 计算预测误差
            yprebuff[j + i * w] = (ypre+255) /radio  # 量化
            img_re[j + i * w] = (yprebuff[j + i * w]-255/radio)*radio+img_re[j + i * w - 1]  # 反量化
            yprebuff[j + i * w] = yprebuff[j + i * w] * radio / 2  # 预测器
            if img_re[j + i * w]>255:
                img_re[j + i * w] = 255
img_re = img_re.astype(np.uint8)
yprebuff = yprebuff.astype(np.uint8)  # 预测误差

# 重建图片
y = y.reshape((h,w))

yprebuff = yprebuff.reshape((h,w))

img_re = img_re.reshape((h,w))

u = u.reshape((h//2,w//2))
v = v.reshape((h//2,w//2))
ru = cv2.resize(u,(w,h))
rv = cv2.resize(v,(w,h))

yvu = cv2.merge((y, rv, ru))
bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR)

yvu_pre = cv2.merge((yprebuff, rv, ru))
bgr_pre = cv2.cvtColor(yvu_pre, cv2.COLOR_YCrCb2BGR)

yvu_re = cv2.merge((img_re, rv, ru))
bgr_re = cv2.cvtColor(yvu_re, cv2.COLOR_YCrCb2BGR)

# 显示结果
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(131), plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
plt.title('原图'), plt.axis('off')
plt.subplot(132), plt.imshow(cv2.cvtColor(bgr_pre, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('预测误差'), plt.axis('off')
plt.subplot(133), plt.imshow(cv2.cvtColor(bgr_re, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('重建'), plt.axis('off')

plt.show()


# psnr和ssim
def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    print("PSNR average SSIM: {}".format(20*math.log10(255.0/rmse)))


def ssim(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)
    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (grayScore, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("gray SSIM: {}".format(grayScore))
    (score0, diffB) = structural_similarity(B1, B2, full=True)
    (score1, diffG) = structural_similarity(G1, G2, full=True)
    (score2, diffR) = structural_similarity(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    print("BGR average SSIM: {}".format(aveScore))


psnr(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),cv2.cvtColor(bgr_re, cv2.COLOR_BGR2RGB))
ssim(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),cv2.cvtColor(bgr_re, cv2.COLOR_BGR2RGB))

