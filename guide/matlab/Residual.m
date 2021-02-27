%% Read Images
[indimage_in, rgbmap_in] = imread('../../input/<Input BMP>');
[indimage_out, rgbmap_out] = imread('../../output/<Output BMP>');

indimage = abs(indimage_in - indimage_out);
image(indimage)
