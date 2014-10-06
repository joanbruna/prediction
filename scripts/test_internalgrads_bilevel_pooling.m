
%% GROUP SPARSITY AND POOLING

ma = 24;
na = 24;

overlapping = 0;
groupsize = 2;
time_groupsize = 2;

if overlapping
    f1 = groupsize/2;
    f2 = time_groupsize/2;
    off1=[1 1 f1+1 f1+1];
    off2=[1 f2+1 1 f2+1];
else
    f1 = groupsize;
    f2 = time_groupsize;
end

box = ones(groupsize,time_groupsize);

[mb,nb] = size(box);
sizeAux = max([ma-max(0,mb-1),na-max(0,nb-1)],0);
aux = zeros(sizeAux);


% test
Z =  rand(ma,na)+0.1;
idx = randperm(ma*na);
Z(idx(1:end/2)) =0;

eps = 0.01;
eps_1 = 1e-6;



Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
P = Poole(1:f1:end,1:f2:end);

% Verify second layer lasso solution
% Z2 = mexLasso(P,Dgn,param0);
% 
% Pr = Dgn * Z2;
% 
% dPr = Dgn'*(Dgn*Z2 - P);
% 
% 
% out = nmf_sol(Z2,Pr,dPr,Dgn,lambda2gn,groupsize,time_groupsize);




if overlapping
    f = 0;
    
    %Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
    for k=1:4
        
        
        tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        fp = zeros(size(tPoole));
        id = find(tPoole>0);
        fp(id) = 1./tPoole(id);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
        uPoole = (conv2(aux,box,'full')).*Z;
        
        f = f + sum(uPoole(:));
 
    end
    
    % perturbation
    dZ = eps_1*randn(size(Z));
    dZ(idx(1:end/2)) =0;
    
    Z_ = Z + dZ;
    
    f_ = 0;
    
    Poole_ = sqrt(conv2(Z_.^2+eps,box,'valid'));
    
    for k=1:4
        
        tPoole_=Poole_(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        fp_ = zeros(size(tPoole_));
        id = find(tPoole_>0);
        fp_(id) = 1./tPoole_(id);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp_;
        uPoole_ = (conv2(aux,box,'full')).*Z_;
        
        f_ = f_ + sum(uPoole_(:));
        
    end
    
    % Compute grads
    dfZ = zeros(size(Z));
    
    tPoole=Poole(1:f1:end,1:f2:end);
    
    fp = zeros(size(tPoole));
    id = find(tPoole>0);
    fp(id) = 1./tPoole(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=fp;
    T1 = conv2(aux,box,'full');
    
    
    Zp = conv2(Z,box,'valid');
    Zp=Zp(1:f1:end,1:f2:end);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=(fp.^3).*Zp;
    T2 = (conv2(aux,box,'full')).*Z;
    
    dfZ = dfZ + T1-T2;
    
%     for k=1:4
%         
%         
%         tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
%         
%         fp = zeros(size(tPoole));
%         id = find(tPoole>0);
%         fp(id) = 1./tPoole(id);
%         
%         aux = 0*aux;
%         aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
%         T1 = conv2(aux,box,'full');
%         
%         
%         Zp = conv2(Z,box,'valid');
%         Zp=Zp(off1(k):groupsize:end,off2(k):time_groupsize:end);
%         
%         aux = 0*aux;
%         aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=(fp.^3).*Zp;
%         T2 = (conv2(aux,box,'full')).*Z;
%         
%         dfZ = dfZ + T1-T2;
%         
%     end
%     
    
else
    
    %Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
    
    Poole=Poole(1:f1:end,1:f2:end);
    
    fp = zeros(size(Poole));
    id = find(Poole>0);
    fp(id) = 1./Poole(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)= fp;
    uPoole = (conv2(aux,box,'full')).*Z;
    
    f = sum(uPoole(:));
    
    % perturbation
    dZ = eps_1*randn(size(Z));
    dZ(idx(1:end/2)) =0;
    
    Z_ = Z + dZ;
    
    Poole_ = sqrt(conv2(Z_.^2+eps,box,'valid'));
    Poole_=Poole_(1:f1:end,1:f2:end);
    
    fp_ = zeros(size(Poole_));
    id = find(Poole_>0);
    fp_(id) = 1./Poole_(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)= fp_;
    uPoole_ = (conv2(aux,box,'full')).*Z_;
    
    f_ = sum(uPoole_(:));
    
    % Compute grads
    
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=fp;
    T1 = conv2(aux,box,'full');
    
    
    Zp = conv2(Z.*dZ,box,'valid');
    Zp=Zp(1:f1:end,1:f2:end);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=(fp.^3).*Zp;
    T2 = (conv2(aux,box,'full')).*Z;
    
    dfZ = T1.*dZ - T2;
    
    
    
    
end

[f_-f, sum(dfZ(:))]/eps_1


break

%% ONLY GROUP SPARSITY


ma = 24;
na = 24;

overlapping = 0;
groupsize = 2;
time_groupsize = 2;

if overlapping
    f1 = groupsize/2;
    f2 = time_groupsize/2;
    off1=[1 1 f1+1 f1+1];
    off2=[1 f2+1 1 f2+1];
else
    f1 = groupsize;
    f2 = time_groupsize;
end

box = ones(groupsize,time_groupsize);

[mb,nb] = size(box);
sizeAux = max([ma-max(0,mb-1),na-max(0,nb-1)],0);
aux = zeros(sizeAux);


% test
Z =  rand(ma,na)+0.1;
idx = randperm(ma*na);
Z(idx(1:end/2)) =0;

eps = 0.01;
eps_1 = 1e-6;

if overlapping
    f = 0;
    
    Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
    for k=1:4
        
        
        tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        fp = zeros(size(tPoole));
        id = find(tPoole>0);
        fp(id) = 1./tPoole(id);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
        uPoole = (conv2(aux,box,'full')).*Z;
        
        f = f + sum(uPoole(:));
        
    end
    
    % perturbation
    dZ = eps_1*randn(size(Z));
    dZ(idx(1:end/2)) =0;
    
    Z_ = Z + dZ;
    
    f_ = 0;
    
    Poole_ = sqrt(conv2(Z_.^2,box,'valid'));
    
    for k=1:4
        
        tPoole_=Poole_(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        fp_ = zeros(size(tPoole_));
        id = find(tPoole_>0);
        fp_(id) = 1./tPoole_(id);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp_;
        uPoole_ = (conv2(aux,box,'full')).*Z_;
        
        f_ = f_ + sum(uPoole_(:));
        
    end
    
    % Compute grads
    dfZ = zeros(size(Z));
    for k=1:4
        
        
        tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        fp = zeros(size(tPoole));
        id = find(tPoole>0);
        fp(id) = 1./tPoole(id);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
        T1 = conv2(aux,box,'full');
        
        
        Zp = conv2(Z,box,'valid');
        Zp=Zp(off1(k):groupsize:end,off2(k):time_groupsize:end);
        
        aux = 0*aux;
        aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=(fp.^3).*Zp;
        T2 = (conv2(aux,box,'full')).*Z;
        
        dfZ = dfZ + T1-T2;
        
    end
    
    
else
    
    Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
    Poole=Poole(1:f1:end,1:f2:end);
    
    fp = zeros(size(Poole));
    id = find(Poole>0);
    fp(id) = 1./Poole(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)= fp;
    uPoole = (conv2(aux,box,'full')).*Z;
    
    f = sum(uPoole(:));
    
    % perturbation
    dZ = eps_1*randn(size(Z));
    dZ(idx(1:end/2)) =0;
    
    Z_ = Z + dZ;
    
    Poole_ = sqrt(conv2(Z_.^2+eps,box,'valid'));
    Poole_=Poole_(1:f1:end,1:f2:end);
    
    fp_ = zeros(size(Poole_));
    id = find(Poole_>0);
    fp_(id) = 1./Poole_(id);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)= fp_;
    uPoole_ = (conv2(aux,box,'full')).*Z_;
    
    f_ = sum(uPoole_(:));
    
    % Compute grads
    
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=fp;
    T1 = conv2(aux,box,'full');
    
    
    Zp = conv2(Z,box,'valid');
    Zp=Zp(1:f1:end,1:f2:end);
    
    aux = 0*aux;
    aux(1:f1:end,1:f2:end)=(fp.^3).*Zp;
    T2 = (conv2(aux,box,'full')).*Z;
    
    dfZ = T1 - T2;
    
end

[f_-f, dfZ(:)'*dZ(:)]/eps_1


