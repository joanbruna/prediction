function out0 = wavelet_transf_gen(in, filts, outsize)
%this function computes a wavelet transform along each row of in


[~,C, N, BS]=size(in);
J = size(filts,2);
out = zeros(C,N*BS,J+1, 'single','gpuArray');

res = squeeze(in);
res = res(:,:);
res = gpuArray(single(res));
out(:,:,1) = res;
res = fft(res);

for j=1:size(filts,2)

out(:,:,j+1) = real(ifft(res.*repmat(filts{j},1,size(res,2))));

end

out = reshape(out,C, N, BS, J+1);
out = permute(out,[1 2 4 3]);
out = reshape(out,C, N*(J+1), BS);

clear res

p=round((C-outsize)/2);

out0(1,:,:,:) = out(p+1:p+outsize,:,:);

