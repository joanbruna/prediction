function [D,W,verbo] = nmf_linear_dynamic_batch(X, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 


renorm=getoptions(options,'renorm_input', 0);
if renorm
    norms = sqrt(sum(X.^2));
    I0=find(norms>0);
    norms=norms(I0);
    X(:,I0)=X(:,I0) ./ repmat(norms,[size(X,1) 1]);
end

% produce_synthesis=getoptions(options,'produce_synthesis',0);

[N,M]=size(X);
K = getoptions(options, 'K', 2*N);
%N: input dimension
%M: number of examples
%K output dimension

% groupsize = getoptions(options,'groupsize',2);

%initial dictionary
batchsize=getoptions(options,'batchsize',128);
II=randperm(floor(M/batchsize)-1);
D=X(:,batchsize*II(1:K));

rho=0;
D=rho*D + (1-rho)*randn(size(D));

D=getoptions(options,'initdictionary',D);
D = mexNormalize(D);

% No dynamics at init.
W = eye(K);


%D(:,2:2:end) = D(:,1:2:end);
rho=0.9;
D=rho*D + (1-rho)*randn(size(D))/sqrt(N);

rho =getoptions(options,'rho',15);


Bd=0*D;
Ad=zeros(size(D,2));

Bw=zeros(size(D,2));
Aw=zeros(size(D,2));

nepochs=getoptions(options,'epochs',4);
%batchsize=getoptions(options,'batchsize',128);
niters=nepochs*M/batchsize;

% use first batch as validation set.
I0 = (II(10)+1):(II(10)+1)+batchsize;
datav=X(:,I0);


%verbose variables
chunks=100;
ch = floor(niters/chunks);


% tot_tested=0;
lambda = getoptions(options,'lambda',0.1);
mu = getoptions(options,'mu',0.5);


rast=1;

% compute initial validation cost 
alpha = nmf_linear_dynamic_pursuit( datav, D, W , options);
rec = D * alpha;
c1 = .5 * norm(rec(:)-datav(:))^2/batchsize;
c2 = lambda * sum(alpha(:))/batchsize;
dyn = W * alpha(:,1:end-1);
alphat = alpha(:,2:end);
c3 = .5 * mu * norm(dyn(:)-alphat(:))^2/batchsize;
currerr = c1 + c2 + c3;
verbo(rast) = currerr;rast=rast+1;
fprintf('current error is %f (%f %f l0=%f) \n', currerr, c1, c2, c3)

for n=1:niters
%update synthesis coefficients
init= mod( n, floor(M/batchsize)); 
I0 = II(1+init):(II(1+init)+batchsize-1);
data=X(:,I0);

%data = X(:,1+init:batchsize+init);

alpha = nmf_linear_dynamic_pursuit( data, D, W , options);

beta = (1-1/n).^rho;

Ad = beta * Ad + (alpha*alpha');
Bd = beta * Bd + (data*alpha');

alphat1 = alpha(:,2:end);
Aw = beta * Aw + (alphat1*alphat1');
Bw = beta * Bw + (alphat1*alpha(:,1:end-1)');


%%dictionary update
D = dictionary_update( D,Ad,Bd,options);

% dynamic matrix update
W = dictionary_update( W,Aw,Bw,options);

if mod(n,ch)==ch-1
fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
%compute error 
% modulus = modphas_decomp(alpha,groupsize);
alpha = nmf_linear_dynamic_pursuit( datav, D, W , options);
rec = D * alpha;
c1 = .5 * norm(rec(:)-datav(:))^2/batchsize;
c2 = lambda * sum(alpha(:))/batchsize;
dyn = W * alpha(:,1:end-1);
alphat = alpha(:,2:end);
c3 = .5 * mu * norm(dyn(:)-alphat(:))^2/batchsize;
currerr = c1 + c2 + c3;
verbo(rast) = currerr;rast=rast+1;
fprintf('current error is %f (%f %f %f) \n', currerr, c1, c2, c3)

% rho=0.025; 
% deadunits=find(diag(A)<rho);
% dead=length(deadunits);
% if dead > 0
% fprintf('%d dead elements out of %d (%f, %f) \n', dead, K, max(diag(A)), min(diag(A)) )
% %Itmp =randperm(M);
% %D(:, deadunits) = X(:,Itmp(1:dead));
% %D = ortho_pools(D',2)';
% end
end
end

% if produce_synthesis
% [~,~,alphas] = time_coeffs_update(D, X, options); %A, B, options, 1, t0);
% %alphas = D'*X;
% modulus = modphas_decomp(alphas,groupsize);
% else
% alphas=0;
% modulus=0;
% end

end


function D= dictionary_update(Din, A,B, options)

iters=getoptions(options,'dict_iters',2);

D=Din;

N=size(B,1);
dia = diag(A)';

tol=1e-6;
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

for i=1:iters

if 0
U = D0 + (B-D0*A).*Att;
nu=sqrt(sum(U.^2));
D0 = U ./ max(1,repmat(nu,[size(U,1) 1]));
else

for j=1:K

u = D0(:,j) + (B(:,j) - D0*(A(:,j)))*At(j);
%non-negative case
u = max(0, u);
D0(:,j) = u / max(1, norm(u));

end
end

end

if fix
D(:,I)=D0;
else
D=D0;
end
 
% D = ortho_pools(D',2)';

end
