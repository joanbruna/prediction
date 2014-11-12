% train model
representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';

gender = 'male';


load(sprintf('%s%s',representation,gender));

imdb.meta.gender = gender;
imdb.meta.NFFT = data.NFFT;
imdb.meta.hop = data.hop;
imdb.meta.fs = data.fs;


batchsize = 100;
n = floor(size(data.X,2)/batchsize);

N = n*batchsize;
A = reshape(data.X(:,1:N),[size(data.X,1),batchsize,1,n]);



B = zeros(size(data.X,1),batchsize,2,n);
B(:,:,1,:) = real(A);
B(:,:,2,:) = imag(A);


imdb.images.set = ones(1,size(B,2));
imdb.images.data = B;

%%
save(sprintf('%simdb_%s',representation,gender),'imdb');