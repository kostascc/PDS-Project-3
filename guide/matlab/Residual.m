%% Read Images
img_in = imread('../../input/<input bitmap>.bmp');
img_out = imread('../../output/<output bitmap>.bmp');

fig = figure;
indimage = 2 .* abs(img_out-img_in);
image(indimage)
saveas(fig,'../../output/RES_<output bitmap>.bmp')
close all;