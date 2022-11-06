clear; clc;
I=rgb2gray(imread('hutao.png'));
subplot(1, 2, 1)
imshow(I);
title('ԭͼ');
T = Otsu(double(I));     %ʹ�ô�򷨼�����ֵ
%disp(['��򷨼���Ҷ���ֵ:', num2str(T)])
BW = imbinarize(I, T/255);
%��ֵ�ָ�
subplot(1, 2, 2)
imshow(BW);
title('��򷨴����');
 
function ThreshValue = Otsu(Imag)
% ��򷨼�����ֵ
% ���룺
%    Imag����ά���飬��ֵ��ʾ�Ҷȣ�
% �����
%    ThreshValue����ֵ
iMax = max(Imag(:));              % ���ֵ
iMin = min(Imag(:));               % ��Сֵ
T = iMin:iMax;                        % �Ҷ�ֵ��Χ
Tval = zeros(size(T));               % ����
[iRow, iCol] = size(Imag);        % ����ά�ȴ�С
imagSize = iRow*iCol;            % ���ص�����
% �����Ҷ�ֵ�����㷽��
for i = 1 : length(T)
    TK = T(i);
    iFg = 0;          % ǰ��
    iBg = 0;          % ����
    FgSum = 0;    % ǰ������
    BgSum = 0;    % ��������
    for j = 1 : iRow
        for k = 1 : iCol
            temp = Imag(j, k);
            if temp > TK
                iFg = iFg + 1;      % ǰ�����ص�ͳ��
                FgSum = FgSum + temp;
            else
                iBg = iBg + 1;      % �������ص�ͳ��
                BgSum = BgSum + temp;
            end
        end
    end
    w0 = iFg/imagSize;      % ǰ������
    w1 = iBg/imagSize;     % ��������
    u0 = FgSum/iFg;         % ǰ���Ҷ�ƽ��ֵ
    u1 = BgSum/iBg;        % �����Ҷ�ƽ��ֵ
    Tval(i) = w0*w1*(u0 - u1)*(u0 - u1);     % ���㷽��
end
[~, flag] = max(Tval);             % ���ֵ�±�
ThreshValue = T(flag);
end