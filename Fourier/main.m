clear all
Gray = imread('./Fig0413Gry.bmp');

FFT = fft2(Gray);%傅里叶变换
FFT_Shift = fftshift(FFT);%对频谱进行移动，是0频率点在中心
AM_Spectrum = log(abs(FFT_Shift));%获得傅里叶变换的幅度谱
%ans = mean(mean(AM_Spectrum));%方式3: 幅度谱均值
%AM_Spectrum(: , :) = ans;%方式3: 幅度谱矩阵赋均值
Phase_Specture = log(angle(FFT_Shift)*180/pi);%获得傅里叶变换的相位谱
%Phase_Specture = 0;%方式2: 相位谱为0
Restructure = ifft2(abs(FFT).*exp(j*(angle(FFT))));%双谱重构
figure(1)
subplot(221)
imshow(Gray)
title('原图像')
subplot(222)
imshow(AM_Spectrum,[])%显示图像的幅度谱，参数'[]'是为了将其值线性拉伸
title('幅度谱')
subplot(223)
imshow(Phase_Specture,[]);
title('相位谱')
subplot(224)
imshow(Restructure,[]);
title('复原图')
