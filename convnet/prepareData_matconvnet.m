
function imdb = prepareData_matconvnet(data,nframes,name,use_single)


imdb.meta.name = name;
imdb.meta.NFFT = data.NFFT;
imdb.meta.hop = data.hop;
imdb.meta.fs = data.fs;

n = floor(size(data.X,2)/nframes);

N = n*nframes;
A = reshape(data.X(:,1:N),[size(data.X,1),nframes,1,n]);
A = permute(A, [3 2 1 4]);

imdb.images.set = ones(1,size(A,4));
imdb.images.data = A;

if use_single
    imdb.images.data = single(imdb.images.data );
end
