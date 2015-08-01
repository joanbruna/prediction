function out = wavelet_transf(in, J)
%this function computes a wavelet transform along each row of in


[~,C, N, BS]=size(in);
out = zeros(N, BS,J+1, 'single','gpuArray');

res = squeeze(in);
res = res(:,:);
for j=1:J

[res, dif] = haar_1d(res);
out((j-1)*N*BS+1:j*N*BS) = reshape(dif(round(size(dif,1)/2),:),N*BS,1);

if j==J
out(J*N*BS+1:(J+1)*N*BS) = reshape(res(round(size(dif,1)/2),:),N*BS,1);
end

end

out = reshape(out, 1, N, BS, J+1);
out = permute(out, [1 4 2 3]);
out = reshape(out, 1, 1, (J+1)*N, BS);
out(1,1,1:N,:) = in(1,ceil(C/2),:,:);

end


function [ave, dif] = haar_1d(res)

h=[1; 1]/sqrt(2);
g=[1; -1]/sqrt(2);
h=gpuArray(single(h));
g=gpuArray(single(g));

ave = conv2(res,h,'same');
dif = conv2(res,g,'same');
ave = ave(1:2:end,:);

end


