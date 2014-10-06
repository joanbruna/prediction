function [lossFun,lossGrad_x,lossGrad_w] = GetLossFun_w(loss_type)


switch lower(loss_type)
    
    case 'hinge'
        %Hinge-loss
        lossFun    = @(x,l,w)mean((1-(w'*x).*l) .* ((w'*x).*l < 1));
        lossGrad_x = @(x,l,w)       - w    *(l  .* ((w'*x).*l < 1));
        lossGrad_w = @(x,l,w)       - x    *(l  .* ((w'*x).*l < 1))';
        
    case 'logit'
        %logistic-loss
        lossFun    = @(x,l,w)mean(  log(1+exp(-(w'*x).*l)));
        lossGrad_x = @(x,l,w)-w * (l ./(1+exp(-(w'*x).*l)) .* exp(-(w'*x).*l));
        lossGrad_w = @(x,l,w)-x * (l ./(1+exp(-(w'*x).*l)) .* exp(-(w'*x).*l))' ;
        
    case 'exp'
        %exponential-loss        
        lossFun    = @(x,l,w)mean(      exp(-(w'*x).*l));
        lossGrad_x = @(x,l,w)-w * (l .* exp(-(w'*x).*l));
        lossGrad_w = @(x,l,w)-x * (l .* exp(-(w'*x).*l))' ;
        
    case 'l2'
        %square-loss        
        lossFun    = @(x,l,w).5 * mean(((w'*x)-l).^2);
        lossGrad_x = @(x,l,w) w *      ((w'*x)-l);
        lossGrad_w = @(x,l,w) x *      ((w'*x)-l)';
                             
        
    otherwise
        error('bad loss type')
        
end