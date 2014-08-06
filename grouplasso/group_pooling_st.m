function [D,D0,verbo,outchunk] = group_pooling_st(X, options)
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
II=randperm(M-1);

%rho=0;
%D=rho*D + (1-rho)*randn(size(D));

D=getoptions(options,'initdictionary',D);

norms = sqrt(sum(D.^2));
D = D./ repmat(norms,[size(D,1) 1]);

%D(:,2:2:end) = D(:,1:2:end);
%rho=0.9;
%D=rho*D + (1-rho)*randn(size(D))/sqrt(N);
%D = ortho_pools(D',2)';

B=0*D;
A=zeros(size(D,2));

nepochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',256);
niters=round(nepochs*M/batchsize);

%verbose variables
chunks=100;
ch = ceil(niters/chunks);

t0 = getoptions(options,'alpha_step',0.25);
t0 = t0 * (1/max(svd(D))^2)
tot_tested=0;
tgroups=getoptions(options,'time_groupsize',2);
options.batchsize=batchsize;
options.time_groupsize=tgroups;
lambda = getoptions(options,'lambda',0.1);

D0=D;
rast=1;
c1=0;
c2=0;


for n=1:niters
%update synthesis coefficients
init= mod( (n-1)*batchsize, M-batchsize+0); 
I0 = II(1+init:batchsize+init-0);
data=zeros(N,options.time_groupsize*length(I0));
for tt=1:options.time_groupsize
data(:,tt:options.time_groupsize:end)=X(:,I0+tt-1);
end
%data = X(:,1+init:batchsize+init);
update_t0=getoptions(options,'update_t0',0);
if mod(n,update_t0)==update_t0-1
t0 = getoptions(options,'alpha_step',0.25);
t0 = t0 * (1/max(svd(D))^2);
end


[A,B,alpha] = time_coeffs_update( D, data, options,A,B,t0, n);
%[A,B,alpha] = time_coeffs_update22( D, data, options,A,B,t0, n);

%measure_cost(alpha, D, data, lambda, groupsize, 'after lasso');

%%dictionary update
D = dictionary_update( D,  A,B,options);


verbo(rast)=measure_cost(alpha, D, data, lambda, groupsize, 'after dict update');
rast=rast+1;

if mod(n,ch)==ch-1
fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
end


if 0
rec = D * alpha;
modulus = modphas_decomp(alpha,groupsize);
c1 = c1 + .5 * norm(rec(:)-data(:))^2/batchsize;
c2 = c2 + lambda * sum(modulus(:))/batchsize;

if mod(n,ch)==ch-1
fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
%compute error 
aux=modulus(:);
c2bis=sum(aux>0)/length(aux);
verbo(rast) = c1+c2;rast=rast+1;
fprintf('current error is %f (%f %f l0=%f) \n', c1+c2, c1, c2, c2bis)
c1=0;
c2=0;

%rho=0.025; 
%deadunits=find(diag(A)<rho);
%dead=length(deadunits);
%if dead > 0
%fprintf('%d dead elements out of %d (%f, %f) \n', dead, K, max(diag(A)), min(diag(A)) )
%Itmp =randperm(M);
%D(:, deadunits) = X(:,Itmp(1:dead));
%D = ortho_pools(D',2)';
%end
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
Ds1 = D(:,1:2:end);
Ds2 = D(:,2:2:end);
corrs = abs(sum(Ds1.*Ds2));
Dtmp = circshift(D,[0 -1]);
Ds1b = Dtmp(:,1:2:end);
Ds2b = Dtmp(:,2:2:end);
corrsb = abs(sum(Ds1b.*Ds2b));
fprintf('dictionary group coherence (even): %f %f %f \n',min(corrs), max(corrs), median(corrs))
fprintf('dictionary group coherence (odd): %f %f %f \n',min(corrsb), max(corrsb), median(corrsb))

end


function out=measure_cost(alpha, D, data, lambda, groupsize, str)

batchsize = size(data,2);
rec = D * alpha;
modulus = modphas_decomp(alpha,groupsize);
c1 = norm(rec(:)-data(:))^2/batchsize;
c2 = lambda * sum(modulus(:))/batchsize;
c3 = sum(modulus(:)>0)/numel(modulus);
out=c1+c2;
fprintf( '%s...%f (%f %f) nonzeros %f \n', str, c1+c2,c1,c2,c3)

end





