function [Dout,ind_out] = sortD(D)

Dout = zeros(size(D));

k = size(D,2);

d = D(:,1);
Dout(:,1) = d;

D = D(:,2:end);

ind_in = 2:k;

ind_out = zeros(1,k);
ind_out(1) = 1;

for i=2:k
    
    id = knnsearch(D',d');
    
    d = D(:,id);
    Dout(:,i) = d;
    
    ind_out(i) = ind_in(id);
    
    if id==1
        D = D(:,2:end);
        ind_in = ind_in(2:end);
        
    elseif id ==size(D,2)
        D = D(:,1:end-1);
        ind_in = ind_in(1:end-1);

    else
        D = [D(:,1:id-1),D(:,id+1:end)];
        ind_in = [ind_in(1:id-1) ind_in(id+1:end)];

    end
    
end

