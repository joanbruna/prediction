

function [Wv,Wn] = nmf_supervised(S1,S2,Wv,Wn,options)



% Initialize
%===============


% Parameters ------------------------------------------------------------
%Niter = 30000;
NiterA = 200;
%tol=0;

weight = 1;
beta2 = 1;
lossFun    = @(Vv,Vn,Wv,Hv,Wn,Hn) betadiv(Vv,Wv*Hv,beta2) + weight*betadiv(Vn,Wn*Hn,beta2);
gradFun_hv    = @(Vv,Vn,Wv,Hv,Wn,Hn) betadiv_grad(Vv,Wv,Hv,beta2,'dH');
gradFun_hn    = @(Vv,Vn,Wv,Hv,Wn,Hn) weight*betadiv_grad(Vn,Wn,Hn,beta2,'dH');
gradFun_wv    = @(Vv,Vn,Wv,Hv,Wn,Hn) betadiv_grad(Vv,Wv,Hv,beta2,'dW'); 
gradFun_wn    = @(Vv,Vn,Wv,Hv,Wn,Hn)weight*betadiv_grad(Vn,Wn,Hn,beta2,'dW');


% Stochastic gradient descent------------------------------------------------------------
eta = 0.1;
sigma = 0;
step0 = 1e-2;
totaliter = 2000;
maxarmijoiter = 5;
bsize = 100;
    rho = 1;

TNv = size(S1,2)-bsize;
TNn = size(S2,2)-bsize;

% Validation data------------------------------------------------------------
SNR_dB = 0;

% Initial dictionaries------------------------------------------------------------
K = size(Wv,2);
nn = size(Wn,2);

%Wv = projectW(dic_v.Wv);
%Wn = projectW(dic_n.Wn);
W = [Wv,Wn];

Wv_init = Wv;
Wn_init = Wn;


% Pursuit parameters ------------------------------------------------------------
beta = 1;
count = 1;

lambda2 = getoptions(options, 'lambda2', 0);


valid = getoptions(options, 'valid', []);
do_valid = 1;
if isempty(valid)
   do_valid = 0;
end


fvmin = inf;
sdrmax = 0;

Mv = zeros(size(Wv));
Mn = zeros(size(Wn));

for iter = 1:totaliter,
    
    fprintf(1, 'iter = %4d \n ', iter);
    fprintf(1,'\n ')
    
    % Validation
    if (~mod(iter,5) || iter==1) && do_valid
        
        
        [~,Hmix] = nmf_admm(abs(valid.X), [Wv,Wn], valid.H_ini, beta, rho,1:2*size(Wv,2));
        
        W1H1 = Wv*Hmix(1:K,:);
        W2H2 = Wn*Hmix(K+1:end,:);
        
        %V_ap = W1H1 +W2H2;
        V_ap = W1H1 +W2H2 + eps;
        
        overlap = valid.overlap;
        l_win = overlap*2;
        T = length(valid.x1);
        
        
        % Reconstruct sources
        SPEECH1 = ((W1H1)./V_ap).*valid.X;
        SPEECH2 = ((W2H2)./V_ap).*valid.X;
        speech1 = cf_istft(SPEECH1,l_win,overlap);
        speech1 = speech1(overlap+1:overlap+T);
        speech2 = cf_istft(SPEECH2,l_win,overlap);
        speech2 = speech2(overlap+1:overlap+T);
        
        Parms =  BSS_EVAL(valid.x1', valid.x2', speech1', speech2', valid.mix');

        r{count} = Parms;
        
        Parms
        
        faux = lossFun(valid.V1,valid.V2,Wv,Hmix(1:K,:),Wn,Hmix((K+1):end,:));
        fv(count) = faux;
        disp(fv(count))
        
        
        save('backup','Wv_init','Wn_init','Wv','Wn','count','fv','r')
        
        count = count + 1;
        
    end
    
    
    % Adjust gradient descent step
    %step = step0*min(1, 50/iter);
    step = step0*min(1, 500/iter);
    
    % Select random patch
    idv = max(round(TNv*rand(1)),1);
    idn = max(round(TNn*rand(1)),1);
    

    Sv = S1(:,idv:(idv+bsize-1));
    
    Sn = S2(:,idn:(idn+bsize-1))*power(10,(-SNR_dB)/20);
    S =  Sv + Sn;
    
    P = abs(S);
    Pv = abs(Sv);
    Pn = abs(Sn);
    

    % Pursuit
    disp('Computing pursuit');
    tic
    %[~,H] = nmf_beta(P,beta, Lambda1, lambda2, Niter, tol, W , 0);
    H_ini = abs(randn(size(W,2),size(P,2))) + 1;
    [~,H] = nmf_admm(P, W, H_ini, beta, rho,1:size(W,2));
    toc
    
    % split activations
    Hv = H(1:K,:);
    Hn = H((K+1):end,:);
    
    % split dictionary
    Wv = W(:,1:K);
    Wn = W(:,(K+1):end);
    
    % Compute the cost function and gradient
    f = lossFun(Pv,Pn,Wv,Hv,Wn,Hn);
    dG = [gradFun_hv(Pv,Pn,Wv,Hv,Wn,Hn); gradFun_hn(Pv,Pn,Wv,Hv,Wn,Hn)];
    
    if f<-1000
        keyboard
    end

    % Compute gradient
    dfW = nmf_beta_grads(H, P, dG, W, lambda2,beta,'dW');
    

    % Compute gradients on W and normalize
    dfW = dfW + [gradFun_wv(Pv,Pn,Wv,Hv,Wn,Hn),gradFun_wn(Pv,Pn,Wv,Hv,Wn,Hn)];
    dfW = dfW/norm( dfW);

    
    % split gradient
    dfWv = dfW(:,1:K);
    dfWn = dfW(:,(K+1):end);
    
    % momentum
    alpha = 0.95;
    Mv_ = alpha*Mv - step*dfWv;
    Mn_ = alpha*Mn - step*dfWn;
    
    
    % Armijo search with projection
    for inneriter = 1:maxarmijoiter,
        disp(['Running armijo ' num2str(inneriter)]);
        tic
        
        Wv_ = max(Wv + Mv_,0);
        Wn_ = max(Wn + Mn_,0);
        
        scale = sum(Wn_,1);
        Wn_ = Wn_ .* repmat(scale.^-1,size(Wn,1),1);
       Hn_ = Hn.* repmat(scale',1,size(Hn,2));
        
        scale = sum(Wv_,1);
        Wv_ = Wv_ .* repmat(scale.^-1,size(Wv_,1),1);
      Hv_ = Hv.* repmat(scale',1,size(Hn,2));

        
        %[~,H_] = nmf_beta(P,beta, Lambda1, lambda2, NiterA, tol, [Wv_,Wn_] , 0,[Hv_;Hn_]);
        [~,H_] = nmf_admm(P, [Wv_,Wn_], H, beta, rho,1:size(W,2),NiterA);
%        H_(H_(:)<0.00000001) = 0;
        
        Hv_ = H_(1:K,:);
        Hn_ = H_((K+1):end,:);
        f_ = lossFun(Pv,Pn,Wv_,Hv_,Wn_,Hn_);
        

        if f_ - f < -sigma*step*norm(dfW(:))^2,
            
            % Only update if there's a decent           
            Mv = Wv_ - Wv;
            Mn = Wn_ - Wn;
            
            Wv = Wv_;
            Wn = Wn_;
            W = [Wv,Wn];
            

            
            H = H_;
            break;
        end
        step = step*eta;
        Mv_ = alpha*Mv - step*dfWv;
        Mn_ = alpha*Mn - step*dfWn;
        toc
    end
    
    mean(sum([Wv,Wn],1))
    
    if abs(f_)>10000 || abs(f)>10000
        keyboard
    end
    fprintf(1, 'Train: f = %10.4f \t Validation: fv = %10.4f \t step = %4.2g (%2d)\n', f_-f, fv(count-1), step, inneriter);
    
end




