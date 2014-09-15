function [lossFun,lossGrad_x,lossGrad_w] = div_beta(beta)


switch beta
    
    case 1
        %Hinge-loss
        lossFun    = @(v,l,w)   mean( v  );
        lossGrad_x = @(x,l,w)       - w    *(l  .* ((w'*x).*l < 1));
        lossGrad_w = @(x,l,w)       - x    *(l  .* ((w'*x).*l < 1))';
        
    case 0
        
        
    otherwise
        error('bad loss type')
        
end