function Hm = precompute_haar(C, J)

Hm = zeros(J+1, C, 'single','gpuArray');
Cm = ceil(C/2);
Hm(1,Cm)=1;
for j=2:J
Hm(j,Cm-2^(j-1)+1:Cm)=-1/sqrt(2^j);
Hm(j,Cm+1:Cm+2^(j-1))=1/sqrt(2^j);

if j==J
Hm(J+1,Cm-2^(J-1)+1:Cm+2^(J-1)) = 1/sqrt(2^J);
end
end

