function  [W1H1,W2H2] = nmf_demix_net(Xn,net)


% net.layers{end}.Ymix = im_mix;
% net.layers{end}.Y1 = im1;
% net.layers{end}.Y2 = im2;

res = [];
res = vl_simplenn(net, Xn, [], res, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

W1H1 = res(end).x(:,:,1,1);
W2H2 = res(end).x(:,:,2,1);


