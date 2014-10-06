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


II=randperm(M-1);
D=X(:,II(1:K));
D=getoptions(options,'initdictionary',D);

norms = sqrt(sum(D.^2));
D = D./ repmat(norms,[size(D,1) 1]);

sort_dict=getoptions(options,'sort_dict',1);
if sort_dict
    D = sortD(D); 
end


pt=getoptions(options,'plot_dict',0);
ptvid=getoptions(options,'plot_viddict',0);
if pt
    figure
    dbimagesc(D+0.001);
    drawnow
end
if ptvid
figure(1);
displayPatches(options.dewhitenMatrix*D);
drawnow
end

%D(:,2:2:end) = D(:,1:2:end);
%rho=0.9;
%D=rho*D + (1-rho)*randn(size(D))/sqrt(N);
%D = ortho_pools(D',2)';

B=0*D;
A=zeros(size(D,2));

Aaux= zeros(size(A));
Baux= zeros(size(B));


nepochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',256);
niters=round(nepochs*M/batchsize);

%initial dictionary
II=randperm(floor(M/batchsize)-0);

%verbose variables
chunks=100;
ch = ceil(niters/chunks);

t0 = getoptions(options,'alpha_step',0.25);
t0 = t0 * (1/max(svd(D))^2);

time_groupsize=getoptions(options,'time_groupsize',2);
options.batchsize=batchsize;
options.time_groupsize=time_groupsize;

% period of dictionary update in iterations
p=getoptions(options,'p',15);

D0=D;
rast=1;

rho = 5;

Aaux= zeros(size(Aaux));
Baux= zeros(size(Baux));
cost = 0;
cost1= 0;
cost2=0;

for n=1:niters
    
    %update synthesis coefficients
    init= mod( n, floor(M/batchsize));
    I0 = II(1+init):(II(1+init)+batchsize-1);
    data=X(:,I0);
    
    
    
    %data = X(:,1+init:batchsize+init);
    update_t0=getoptions(options,'update_t0',0);
    if mod(n,update_t0)==update_t0-1
        t0 = getoptions(options,'alpha_step',0.25);
        t0 = t0 * (1/max(svd(D))^2);
    end
    
    
    % [alpha,cost_aux] = time_coeffs_update( D, data, options,t0);
    [alpha,cost_aux] = group_pooling_semisup( D, data, options,t0);
    
    
    cost = cost + cost_aux.tot;
    cost1 = cost1 + cost_aux.c1;
    cost2 = cost2 + cost_aux.c2;
    
    aux = (alpha*alpha');
    Aaux = Aaux + aux;
    Baux = Baux + (data*alpha');

    
    % update the dictionary every p mini-batches
    if mod(n,p)==p-1
        
        beta = (1-(p-1)/n).^rho;
        
        A = beta * A + 1/p/batchsize*Aaux;
        B = beta * B + 1/p/batchsize*Baux;
        
        D = dictionary_update( D,  A,B,options);
        
        Aaux= zeros(size(Aaux));
        Baux= zeros(size(Baux));

        
        rast=rast+1;
        
        if mod(n,ch)==ch-1
            fprintf('done chunk %d of %d\n',ceil(n/ch),chunks )
        end
        
        fprintf('It %d of %d; Costs (%f %f %f) \n', n, niters, cost, cost1, cost2 );
        cost = 0;     
        cost1 = 0;     
        cost2 = 0;     
        
        if pt
        figure
        dbimagesc(D+0.001);
        drawnow
        end

	if ptvid & mod(n,8*p)==p-1
	figure(1);
	displayPatches(options.dewhitenMatrix*D);
	figure(2);
	imagesc(alpha);
	drawnow
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


