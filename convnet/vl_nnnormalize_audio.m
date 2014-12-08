function varargout = vl_nnnormalize_audio(X,param,dzdy)
%    Y = VL_FILTERMASK(X, F, B) 
%
%
%

if nargin <= 2
    dzdy = [];
end

% no division by zero


N = 2; % not implemented for more than 2 channels
kappa = param(2);
alpha = 1;
beta = 1;

DEN = 0*X;
DEN(:,:,1:2:end,:) = (kappa + (X(:,:,1:2:end,:).^2 + X(:,:,2:2:end,:).^2 ));
DEN(:,:,2:2:end,:) = DEN(:,:,1:2:end,:);

if isempty(dzdy)
    Z = (X.^2)./DEN; 
else 
    Z = 0*X;
 
    AUX = dzdy(:,:,1:2:end,:) - dzdy(:,:,2:2:end,:);
    Z(:,:,1:2:end,:) = AUX.*( 2*X(:,:,1:2:end,:).*(kappa + X(:,:,2:2:end,:).^2 ) )./(DEN(:,:,1:2:end,:).^2 );
    
    Z(:,:,2:2:end,:) = -AUX.*( 2*X(:,:,2:2:end,:).*(kappa + X(:,:,1:2:end,:).^2 ) )./(DEN(:,:,2:2:end,:).^2 );
    
%     GM = ( p*X(:,:,1,:).^(p-1).*( X(:,:,2,:).^p + eps)./(DEN.^2 ));
%     Z(:,:,1,:) =  dzdy(:,:,1,:).*GM - dzdy(:,:,2,:).*GM;
%     
%     GM = ( p*X(:,:,2,:).^(p-1).*( X(:,:,1,:).^p + eps)./(DEN.^2 ));
%     Z(:,:,2,:) =  -dzdy(:,:,1,:).*GM + dzdy(:,:,2,:).*GM;
end


varargout{1} = Z;

