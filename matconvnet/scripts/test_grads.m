

% Load MNIST data
imdb = load('data/mnist-baseline/imdb');

% Load the network
load('data/mnist-baseline/net-epoch-100');


idx = randperm(numel(imdb.images.labels));
batch = idx(1:500);


% test gradient
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

im_ = [];
labels_ = [];
for i = 1:10
ii = find(labels == i);
im_ = cat(4,im_,im(:,:,:,ii));
labels_ = [labels_ labels(ii)];
end

im = im_;
labels = labels_;

net.layers{end}.class = labels;

%% B-prop
res = [];
res_bp = vl_simplenn(net, im, single(1), res, ...
    'disableDropout', true, ...
    'conserveMemory', 0, ...
    'sync', 1) ;

% change a filter

epsilon = 1e-4;

net_a = net; net_b = net;

l = 5;
f = 3;
w = 1;
h = 3;

net_a.layers{l}.filters(h,w,1,f) =  net_a.layers{l}.filters(h,w,1,f) + epsilon;
net_b.layers{l}.filters(h,w,1,f) =  net_b.layers{l}.filters(h,w,1,f) - epsilon;

res_a = vl_simplenn(net_a, im, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_a = res_a(end).x;

res_b = vl_simplenn(net_b, im, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_b = res_b(end).x;
grad_num = (obj_a-obj_b)/(2*epsilon);

grad = res_bp(l).dzdw{1}(h,w,1,f);

[ grad_num, grad abs(grad - grad_num)/abs(grad)*100]



