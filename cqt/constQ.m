function [A,t] = constQ(y, W, sstep)

Nwin  = size(W,2);                % Window size
Y = buffer(y, Nwin, Nwin-sstep);
t=-(Nwin+1)/2+sstep*[1:size(Y,2)];

idx = find(t>=0);
t = t(idx);
A = abs(W*Y(:,idx));


