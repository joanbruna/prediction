% --------------------------------------------------------------------
function [im, im_mix,im1,im2] = getBatch_nmf(imdb1, imdb2, batch,batch2,epsilon)
% --------------------------------------------------------------------
im1 = imdb1.images.data(:,:,:,batch) ;
im2 = imdb2.images.data(:,:,:,batch2) ;

im_mix = im1 + im2;

im = abs(im_mix);
im = softNormalize(im,epsilon,3);

tc = size(im,2);
if tc > 1
    im_mix = im_mix(:,ceil(tc/2),:,:);
    im2 = im2(:,ceil(tc/2),:,:);
    im1 = im1(:,ceil(tc/2),:,:);
end

