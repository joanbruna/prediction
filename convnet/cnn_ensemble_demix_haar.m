function  [W1H1,W2H2] = cnn_ensemble_demix_haar(Xn,net,J,epsilon)



res = [];
N = size(Xn,1);

Xn = permute(Xn, [3 2 1 4]);

Xn = gpuArray(Xn);

if nargin > 2
%keyboard
Xn = wavelet_transf_batch(Xn, J);
%tmp = sqrt(sum(Xn.^2,3));
%Xn = Xn./repmat(epsilon + tmp, [1 1 size(Xn,3) 1]) ;
%J=0;
end

%keyboard

res = vl_simplenn(net, single(Xn), [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

Yn = permute(res(end).x, [3 2 1]);


W1H1 = gather( Yn(1:2:end,:) );
W2H2 = gather( Yn(2:2:end,:) );
