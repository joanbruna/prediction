% --------------------------------------------------------------------
function [im, im_mix,im1,im2] = getBatch_nmf_debug(imdb1, imdb2, batch,batch2,epsilon,C)
% --------------------------------------------------------------------

N = length(batch);
im1 = imdb1.images.data(:,:,:,batch) ;
im1 = permute(im1,[3,4,1,2]);
im1 = reshape(im1,[size(im1,1),C,1,N/C]);
im1 = permute(im1,[3,2,1,4]);
im2 = imdb2.images.data(:,:,:,batch2) ;
im2 = permute(im2,[3,4,1,2]);
im2 = reshape(im2,[size(im2,1),C,1,N/C]);
im2 = permute(im2,[3,2,1,4]);

im_mix = im1 + im2;

im = abs(im_mix);
im = softNormalize(im,epsilon,3);

im1 = single(im1);
im2 = single(im2);
im = single(im);


