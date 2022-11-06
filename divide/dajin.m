clear; clc;
I=rgb2gray(imread('hutao.png'));
subplot(1, 2, 1)
imshow(I);
title('原图');
T = Otsu(double(I));     %使用大津法计算阈值
%disp(['大津法计算灰度阈值:', num2str(T)])
BW = imbinarize(I, T/255);
%阈值分割
subplot(1, 2, 2)
imshow(BW);
title('大津法处理后');
 
function ThreshValue = Otsu(Imag)
% 大津法计算阈值
% 输入：
%    Imag：二维数组，数值表示灰度；
% 输出：
%    ThreshValue：阈值
iMax = max(Imag(:));              % 最大值
iMin = min(Imag(:));               % 最小值
T = iMin:iMax;                        % 灰度值范围
Tval = zeros(size(T));               % 方差
[iRow, iCol] = size(Imag);        % 数据维度大小
imagSize = iRow*iCol;            % 像素点数量
% 遍历灰度值，计算方差
for i = 1 : length(T)
    TK = T(i);
    iFg = 0;          % 前景
    iBg = 0;          % 背景
    FgSum = 0;    % 前景总数
    BgSum = 0;    % 背景总数
    for j = 1 : iRow
        for k = 1 : iCol
            temp = Imag(j, k);
            if temp > TK
                iFg = iFg + 1;      % 前景像素点统计
                FgSum = FgSum + temp;
            else
                iBg = iBg + 1;      % 背景像素点统计
                BgSum = BgSum + temp;
            end
        end
    end
    w0 = iFg/imagSize;      % 前景比例
    w1 = iBg/imagSize;     % 背景比例
    u0 = FgSum/iFg;         % 前景灰度平均值
    u1 = BgSum/iBg;        % 背景灰度平均值
    Tval(i) = w0*w1*(u0 - u1)*(u0 - u1);     % 计算方差
end
[~, flag] = max(Tval);             % 最大值下标
ThreshValue = T(flag);
end