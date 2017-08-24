% Use the following ffmpeg command in command prompt to separate audio
% files:
% ffmpeg -i input-video.mp4 -vn -acodec copy output-audio.mp4
% You must have ffmpeg installed.

[y_1,Fs_1] = audioread('output_left.mp4');
[y_2,Fs_2] = audioread('output_right.mp4');
y_1_first = y_1(:,1);
y_2_first = y_2(:,1);
D = finddelay(y_1_first, y_2_first);

% You can look up the audio sample rate in the .mp4 file properties.
audio_sample_rate = 48000;
offset_seconds = D / audio_sample_rate;
