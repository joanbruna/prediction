function [D,W,verbo] = nmf_linear_dynamic(X, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 


renorm=getoptions(options,'renorm_input', 0);
if renorm
    th = 1;
    norms = sqrt(sum(X.^2));
    I0=find(norms>0);
    norms=norms(I0)+th;
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
batchsize=getoptions(options,'batchsize',100);
p=getoptions(options,'p',15);
II=randperm(floor(M/batchsize)-0);
idx = randperm(M);
D=X(:,idx(1:K));

D=getoptions(options,'initdictionary',D);
D = mexNormalize(D);

% No dynamics at init.
W = eye(K);


rho =getoptions(options,'rho',3);


Bd=0*D;
Ad=zeros(size(D,2));

Bw=zeros(size(D,2));
Aw=zeros(size(D,2));

Adaux= zeros(size(Ad));
Bdaux= zeros(size(Bd));

Awaux= zeros(size(Aw));
Bwaux= zeros(size(Bw));

nepochs=getoptions(options,'epochs',4);
%batchsize=getoptions(options,'batchsize',128);
niters=nepochs*floor(M/batchsize);

% use first batch as validation set.
I0 = (II(10)+1):(II(10)+1)+batchsize;
datav=X(:,I0);


%verbose variables
chunks=100;
ch = floor(niters/chunks);


% tot_tested=0;
lambda = getoptions(options,'lambda',0.1);
mu = getoptions(options,'mu',0.5);

D0=D;
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
fprintf('current error is %f (%f %f %f) \n', currerr, c1, c2, c3)



for n=1:niters
%update synthesis coefficients
init= mod( n, floor(M/batchsize)); 
I0 = II(1+init):(II(1+init)+batchsize-1);
data=X(:,I0);

%data = X(:,1+init:batchsize+init);

alpha = nmf_linear_dynamic_pursuit( data, D, W , options);

aux = (alpha*alpha');
Adaux = Adaux + aux; 
Bdaux = Bdaux + (data*alpha');

Awaux = Awaux + (aux - alpha(:,1)*alpha(:,1)');
Bwaux = Bwaux + (alphat(:,2:end)*alpha(:,1:end-1)');

% update the dictionary every p mini-batches
if mod(n,p)==p-1

beta = (1-(p-1)/n).^rho;

Ad = beta * Ad + 1/p/batchsize*Adaux;
Bd = beta * Bd + 1/p/batchsize*Bdaux;

Aw = beta * Aw + 1/p/batchsize*Awaux;
Bw = beta * Bw + 1/p/batchsize*Bwaux;



%%dictionary update
D = dictionary_update( D,Ad,Bd,options);

% dynamic matrix update
W = dictionary_update( W,Aw,Bw,options);


Adaux= zeros(size(Adaux));
Bdaux= zeros(size(Bdaux));
f
Awaux= zeros(size(Awaux));
Bwaux= zeros(size(Bwaux));

% compute validation after every dic_update
%
% end
% 
% if mod(n,ch)==ch-1
fprintf('done chunk %f of %f\n',ceil(n/p),floor(niters/p) )
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


save temp D W options n

end

end



end


function D= dictionary_update(Din, A,B, options)

iters=getoptions(options,'dict_iters',2);

D=Din;

N=size(B,1);
dia = diag(A)';

tol=1e-9;
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

if min(D(:))<0
    keyboard
end

% D = ortho_pools(D',2)';

end


