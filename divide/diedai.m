A = rgb2gray(imread('hutao.png'));
figure;
subplot(1,2,1);
imshow(A);
title('ԭͼ')
T = mean2(A);   %ȡ��ֵ��Ϊ��ʼ��ֵ
done = false;   %��������ѭ������
i = 0;
% whileѭ�����е���
while ~done
    r1 = find(A<=T);  %С����ֵ�Ĳ���
    r2 = find(A>T);   %������ֵ�Ĳ���
    Tnew = (mean(A(r1)) + mean(A(r2))) / 2;  %����ָ�������ֵ���ֵ��ֵ�ľ�ֵ
    done = abs(Tnew - T) < 1;     %�жϵ����Ƿ�����
    T = Tnew;      %�粻����,�򽫷ָ��ľ�ֵ�ľ�ֵ��Ϊ�µ���ֵ����ѭ������
    i = i+1;
end
A(r1) = 0;   %��С����ֵ�Ĳ��ָ�ֵΪ0
A(r2) = 1;   %��������ֵ�Ĳ��ָ�ֵΪ1   �������ǽ�ͼ��ת���ɶ�ֵͼ��

subplot(1,2,2);
imshow(A,[]);
title('���������')