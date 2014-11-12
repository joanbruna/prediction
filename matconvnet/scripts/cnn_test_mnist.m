

function info = cnn_test_mnist(net, imdb)

ii = find(imdb.images.set == 3);
n = length(ii);

opts.batchSize = 500;
opts.errorType = 'multiclass';

info.objective = 0 ;
info.error = 0;
info.topFiveError = 0;

for i= 1:opts.batchSize:n
    
    % test gradient
    batch =i:min(i+opts.batchSize-1, n);
    
    fprintf('validation: processing batch %3d of %3d ...', ...
        fix(i/opts.batchSize)+1, ceil(n/opts.batchSize)) ;
    
    
    im = imdb.images.data(:,:,:,ii(batch));
    labels = imdb.images.labels(1,ii(batch));
    
    net.layers{end}.class = labels;
    
    res = [];
    res = vl_simplenn(net, im, [], res, ...
        'disableDropout', true, ...
        'conserveMemory', 1, ...
        'sync', 1) ;
    
    info = updateError(opts, info, net, res);
    
    fprintf(' err %.1f err5 %.1f', ...
        info.error/n*100, info.topFiveError(end)/n*100) ;
        fprintf('\n') ;
    
end


% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
end