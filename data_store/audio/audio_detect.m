[y_1,Fs_1] = audioread('left.mp4');
[y_2,Fs_2] = audioread('right.mp4');
y_1_first = y_1(:,1);
y_2_first = y_2(:,1);
D = finddelay(y_1_first, y_2_first);


