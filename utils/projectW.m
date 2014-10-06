function W = projectW(W)


W = max(W,0);

for i=1:size(W,2)
    n = norm(W(:,i));
    if n>1
        W(:,i) = W(:,i)/n;
    end
end