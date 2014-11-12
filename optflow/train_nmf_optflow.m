function [D,D0] = train_nmf_optflow(X, options)
%this function performs a dictionary learning using
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2013 Courant Institute

renorm=getoptions(options,'renorm_input', 0);
if renorm
    norms = sqrt(sum(X.^2));
    I0=find(norms>0);
    norms=norms(I0);
    X(:,I0)=X(:,I0) ./ repmat(norms,[size(X,1) 1]);
end


[N,M]=size(X);
K = getoptions(options, 'K', 2*N);
lambda = getoptions(options,'lambda', 0.1);


init_rand = getoptions(options, 'init_rand', 0);
init_nmf = getoptions(options, 'init_nmf', 1);

if init_nmf
    
    if init_rand == 1
    init_rand = 0;
    warning('More than one initialization method set to 1. Using NMF.')
    end
    
    param0 = struct;
    param0.K = K;
    param0.lambda = lambda;
    param0.posD = 1;
    param0.posAlpha = 1;
    param0.iter = 200;
    param0.D = max(1+randn(N,K),0) + 0.1;
    
    D = mexTrainDL(X, param0);
elseif init_rand
    
    D = max(1+randn(N,K),0) + 0.1;
    
else
    
    II=randperm(M-1);
    D=X(:,II(1:K));
end
D=getoptions(options,'initdictionary',D);

norms = sqrt(sum(D.^2));
D = D./ repmat(norms,[size(D,1) 1]);

sort_dict=getoptions(options,'sort_dict',1);
if sort_dict
    D = sortD(D); 
end


use_flow=getoptions(options,'use_flow',1);


pt=getoptions(options,'plot_dict',0);
if pt
    figure
    dbimagesc(D+0.001);
    drawnow
end


B=0*D;
A=zeros(size(D,2));

Aaux= zeros(size(A));
Baux= zeros(size(B));


nepochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',256);
niters=round(nepochs*M/batchsize);

%initial dictionary
II=randperm(floor(M/batchsize)-1);

%verbose variables
chunks=100;
ch = ceil(niters/chunks);

%t0 = getoptions(options,'alpha_step',0.25);
%t0 = t0 * (1/max(svd(D))^2);


% period of dictionary update in iterations
p=getoptions(options,'p',5);
iter_theta = getoptions(options,'iter_theta',1);

% if no optflow is used, theta should be 1
if ~use_flow
    iter_theta = 1;
end

D0=D;
rast=1;

rho = 5;

Aaux= zeros(size(Aaux));
Baux= zeros(size(Baux));
cost = 0;


ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.1;
ptheta.lambdar = 0.00001;

options.lambda_t = ptheta.lambda;
options.lambda_tr = ptheta.lambdar;
options.hn = ptheta.hn;
options.sigma = ptheta.sigma;

if M<=batchsize
    data = X;
    p = 1;
    beta = 0;
    niters = 20;
end

for n=1:niters
    
    %update synthesis coefficients
    if M>batchsize
    init= mod( n, floor(M/batchsize)-1);
    init_batch = batchsize*II(1+init);
    I0 = init_batch:(init_batch+batchsize-1);
    data=X(:,I0);
    end
    
    %data = X(:,1+init:batchsize+init);
%     update_t0=getoptions(options,'update_t0',0);
%     if mod(n,update_t0)==update_t0-1
%         t0 = getoptions(options,'alpha_step',0.25);
%         t0 = t0 * (1/max(svd(D))^2);
%     end
    
    
    % [alpha,cost_aux] = time_coeffs_update( D, data, options,t0);
    alpha = zeros(K,size(data,2));
    theta = alpha;
    for j = 1:3
        
        [alpha,cost_aux,Salpha] = nmf_optflow( data, D, theta, options);

        if use_flow
        %theta = optflow_taylor2(alpha, ptheta,theta);
        theta = optflow_taylor_temp(alpha, ptheta);
        end
        
    end

    cost = cost + cost_aux;
    
    aux = (alpha*alpha');
    Aaux = Aaux + aux;
    Baux = Baux + (data*alpha');

    
%    update the dictionary every p mini-batches
    if mod(n,p)==p-1
        
        if M>batchsize
            beta = (1-(p-1)/n).^rho;
        end
        
        A = beta * A + 1/p/batchsize*Aaux;
        B = beta * B + 1/p/batchsize*Baux;
        
        D = dictionary_update( D,  A,B,options);
        
        Aaux= zeros(size(Aaux));
        Baux= zeros(size(Baux));

        
        rast=rast+1;
        
        if mod(n,ch)==ch-1
            fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
        end
        
        fprintf('Average costs %f \n', cost/p );
        cost = 0;     
        
        if 1
        figure(3)
%         dbimagesc(D+0.001);
        subplot(311)
        imagesc(alpha)
        subplot(312)
        imagesc(Salpha)
        subplot(313)
        dbimagesc(data+0.001);
        drawnow
        end
        
        %save temp_dic D options 
        
    end
    
end



end


function D= dictionary_update(Din, A,B,options)

iters=getoptions(options,'dict_iters',2);
nmf=getoptions(options,'nmf', 0);

D=Din;

N=size(B,1);
dia = diag(A)';

%lr=1e-2;
tol=1e-8;
I=find(dia>tol);
fix=0;

if length(I) < length(dia)
    
    dia=dia(I);
    D0=D(:,I);
    B=B(:,I);
    A=A(I,I);
    fix=1;
else
    D0=D;
end

At=(dia.^(-1));
Att=repmat(At,[size(B,1) 1]);

K=size(D0,2);
Ip = randperm(K);

for i=1:iters
    
    for j=1:K
        
        u = D0(:,Ip(j)) + (B(:,Ip(j)) - D0*(A(:,Ip(j))))*At(Ip(j));
        if nmf
            u = max(0,u);
        end
        D0(:,Ip(j)) = u / max(1, norm(u));
        
    end
end

if fix
    D(:,I)=D0;
else
    D=D0;
end

%D = ortho_pools(D',2)';
% Ds1 = D(:,1:2:end);
% Ds2 = D(:,2:2:end);
% corrs = abs(sum(Ds1.*Ds2));
% Dtmp = circshift(D,[0 -1]);
% Ds1b = Dtmp(:,1:2:end);
% Ds2b = Dtmp(:,2:2:end);
% corrsb = abs(sum(Ds1b.*Ds2b));
% fprintf('dictionary group coherence (even): %f %f %f \n',min(corrs), max(corrs), median(corrs))
% fprintf('dictionary group coherence (odd): %f %f %f \n',min(corrsb), max(corrsb), median(corrsb))

end


