function  [W1H1,W2H2] = dnn_demix(Xn,net)


res = [];
C = net.layers{4}.C;

W1H1 = 0*Xn;
W2H2 = 0*Xn;

N = floor(size(Xn,2)/C)*C;
for i=1:C:N
    res = vl_simplenn(net, single(Xn(:,i:i+(C-1))), [], res, ...
        'disableDropout', true, ...
        'conserveMemory', 1, ...
        'sync', 1) ;
    
    W1H1(:,i:i+(C-1)) = res(end).x(:,:,1);
    W2H2(:,i:i+(C-1)) = res(end).x(:,:,2);
    
end


if nargout>2
   H1 = H(1:k,:);
   H2 = H(k+1:end,:);
end

