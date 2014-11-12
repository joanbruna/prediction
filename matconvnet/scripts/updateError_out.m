% -------------------------------------------------------------------------
function info = updateError_out(opts, info, net, res)
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