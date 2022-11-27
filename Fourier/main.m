clear all
Gray = imread('./Fig0413Gry.bmp');

FFT = fft2(Gray);%����Ҷ�任
FFT_Shift = fftshift(FFT);%��Ƶ�׽����ƶ�����0Ƶ�ʵ�������
AM_Spectrum = log(abs(FFT_Shift));%��ø���Ҷ�任�ķ�����
%ans = mean(mean(AM_Spectrum));%��ʽ3: �����׾�ֵ
%AM_Spectrum(: , :) = ans;%��ʽ3: �����׾��󸳾�ֵ
Phase_Specture = log(angle(FFT_Shift)*180/pi);%��ø���Ҷ�任����λ��
%Phase_Specture = 0;%��ʽ2: ��λ��Ϊ0
Restructure = ifft2(abs(FFT).*exp(j*(angle(FFT))));%˫���ع�
figure(1)
subplot(221)
imshow(Gray)
title('ԭͼ��')
subplot(222)
imshow(AM_Spectrum,[])%��ʾͼ��ķ����ף�����'[]'��Ϊ�˽���ֵ��������
title('������')
subplot(223)
imshow(Phase_Specture,[]);
title('��λ��')
subplot(224)
imshow(Restructure,[]);
title('��ԭͼ')
