
function imdb = prepareData_matconvnet(data,batchsize,name,use_single)



imdb.meta.name = name;
imdb.meta.NFFT = data.NFFT;
imdb.meta.hop = data.hop;
imdb.meta.fs = data.fs;


n = floor(size(data.X,2)/batchsize);

N = n*batchsize;
A = reshape(data.X(:,1:N),[size(data.X,1),batchsize,1,n]);



B = zeros(size(data.X,1),batchsize,2,n);
B(:,:,1,:) = real(A);
B(:,:,2,:) = imag(A);


imdb.images.set = ones(1,size(B,4));
imdb.images.data = B;

if use_single
    imdb.images.data = single(imdb.images.data );
end