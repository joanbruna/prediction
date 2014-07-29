function [D,D0,verbo,outchunk] = nn_group_pooling_st(X, options)
%this function performs a dictionary learning using 
%the proximal toolbox and iterated gradient descent
%from Mairal et Al (2010)
%requires the spams proximal operator toolbox 

%we learn a dictionary which maximizes group sparsity,
%where groups are bi-clusters.

%Joan Bruna 2013 Courant Institute

renorm=getoptions(options,'renorm_input', 1);
if renorm
norms = sqrt(sum(X.^2));
I0=find(norms>0);
norms=norms(I0);
X(:,I0)=X(:,I0) ./ repmat(norms,[size(X,1) 1]);
end

produce_synthesis=getoptions(options,'produce_synthesis',0);

[N,M]=size(X);
K = getoptions(options, 'K', 2*N);
%N: input dimension
%M: number of examples
%K output dimension

groupsize = getoptions(options,'groupsize',2);

%initial dictionary
II=randperm(M-1);
D=X(:,II(1:K));

%rho=0;
%D=rho*D + (1-rho)*randn(size(D));

D=getoptions(options,'initdictionary',D);

norms = sqrt(sum(D.^2));
D = D./ repmat(norms,[size(D,1) 1]);

%D(:,2:2:end) = D(:,1:2:end);
rho=0.9;
D=rho*D + (1-rho)*randn(size(D))/sqrt(N);

B=0*D;
A=zeros(size(D,2));

nepochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',128);
niters=nepochs*M/batchsize;

%verbose variables
chunks=100;
ch = floor(niters/chunks);

t0 = getoptions(options,'alpha_step',0.25);
t0 = t0 * (1/max(svd(D))^2)
tot_tested=0;
tgroups=getoptions(options,'time_groupsize',2);
options.batchsize=batchsize;
options.time_groupsize=tgroups;
lambda = getoptions(options,'lambda',0.1);

D0=D;
rast=1;

for n=1:niters
%update synthesis coefficients
init= mod( (n-1)*batchsize, M-batchsize+1); 
I0 = II(1+init:batchsize+init-1);
I1 = I0+1;
data=zeros(N,2*length(I0));
data(:,1:2:end)=X(:,I0);
data(:,2:2:end)=X(:,I1);
%data = X(:,1+init:batchsize+init);
update_t0=getoptions(options,'update_t0',32);
if mod(n,update_t0)==update_t0-1
t0 = getoptions(options,'alpha_step',0.25);
t0 = t0 * (1/max(svd(D))^2);
end
[A,B,alpha] = nn_time_coeffs_update( D, data, options,A,B,t0);

%%dictionary update
D = dictionary_update( D,  A,B,options);


if mod(n,ch)==ch-1
fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
%compute error 
modulus = modphas_decomp(alpha,groupsize);
rec = D * alpha;
c1 = .5 * norm(rec(:)-data(:))^2/batchsize;
c2 = lambda * sum(modulus(:))/batchsize;
aux=modulus(:);
c2bis=sum(aux>0)/length(aux);
currerr = c1 + c2;
verbo(rast) = currerr;rast=rast+1;
fprintf('current error is %f (%f %f l0=%f) \n', currerr, c1, c2, c2bis)

rho=0.025; 
deadunits=find(diag(A)<rho);
dead=length(deadunits);
if dead > 0
fprintf('%d dead elements out of %d (%f, %f) \n', dead, K, max(diag(A)), min(diag(A)) )
%Itmp =randperm(M);
%D(:, deadunits) = X(:,Itmp(1:dead));
%D = ortho_pools(D',2)';
end
end
end

if produce_synthesis
[~,~,alphas] = time_coeffs_update(D, X, options); %A, B, options, 1, t0);
%alphas = D'*X;
modulus = modphas_decomp(alphas,groupsize);
else
alphas=0;
modulus=0;
end

end


function D= dictionary_update(Din, A,B,options)

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
 
%D = ortho_pools(D',2)';

end


