% --------------------------------------------------------------------
function [im, im_mix,im1,im2] = getBatch_nmf_single(imdb1, imdb2, batch,batch2,epsilon)
% --------------------------------------------------------------------
im1_aux = imdb1.images.data(:,:,:,batch) ;
im2_aux = imdb2.images.data(:,:,:,batch2) ;

M = 10;
im_mix = cat(4,im1_aux(:,:,:,1:M)+im2_aux(:,:,:,1:M),im1_aux(:,:,:,M+1:end),im2_aux(:,:,:,M+1:end));

im1 = cat(4,im1_aux(:,:,:,1:M),im1_aux(:,:,:,M+1:end),zeros(size(im2_aux(:,:,:,M+1:end))));
im2 = cat(4,im2_aux(:,:,:,1:M),zeros(size(im1_aux(:,:,:,M+1:end))),im2_aux(:,:,:,M+1:end));

im = sqrt(im_mix(:,:,1,:).^2 + im_mix(:,:,2,:).^2);
im = softNormalize(im,epsilon);

im1 = single(im1);
im2 = single(im2);
im = single(im);

