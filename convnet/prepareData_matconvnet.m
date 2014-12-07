
function imdb = prepareData_matconvnet(data,nframes,name,use_single,is_stft)


if ~exist('is_stft','var')
    is_stft = 1;
end

imdb.meta.name = name;
if is_stft
imdb.meta.NFFT = data.NFFT;
imdb.meta.hop = data.hop;
imdb.meta.fs = data.fs;
end

n = floor(size(data.X,2)/nframes);

N = n*nframes;
data.X = circshift(data.X, [0 round(nframes*rand)]);
A = reshape(data.X(:,1:N),[size(data.X,1),nframes,1,n]);
A = permute(A, [3 2 1 4]);

imdb.images.set = ones(1,size(A,4));
imdb.images.data = A;

if use_single
    imdb.images.data = single(imdb.images.data );
end
