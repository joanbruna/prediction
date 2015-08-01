function Hm = precompute_haar_3conv(C, J)

Hm = zeros(3*(J+0), C, 'single','gpuArray');
Cm = ceil(C/2);
%Hm(1,Cm)=1;
for j=1:J
jsf=floor(2^(j-2));
jsc=ceil(2^(j-2));
jn = 2^(j-1);
Hm(3*(j-1)+1,Cm-jsc-jn+1:Cm-jsc)=1/sqrt(2^j);
Hm(3*(j-1)+2,Cm-jsc+1:Cm+jsf)=1/sqrt(2^j);
Hm(3*(j-1)+3,Cm+jsf+1:Cm+jsf+jn)=1/sqrt(2^j);
%Hm(j,Cm-2^(j-1)+1:Cm)=-1/sqrt(2^j);
%Hm(j,Cm+1:Cm+2^(j-1))=1/sqrt(2^j);
end

