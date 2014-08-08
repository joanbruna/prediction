nparam = struct;
rep = 10;
rates = zeros(rep,3);
obj = zeros(rep,1);


K = size(D,2);

for i=1:rep

nparam.Kn=2; %
nparam.iter=200; 
nparam.pos=1;
nparam.lambda = param.lambda;
nparam.verbose = 0;


Pmix = mexNormalize(Vmix);
%Pmix = Vmix ./ repmat(sqrt(epsilon^2+sum(Vmix.^2)),size(Vmix,1),1) ;
[H,Wn,obj(i)] = nmf_beta(Pmix,D,nparam);


Hs = H(1:K,:);
Hn = H((K+1):end,:);


R = {};
R{1} = D* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

if isnan(SDR(1))
    keyboard
end

rates(i,:) = [SDR(1) SIR(1) SAR(1)];


end

disp(mean(rates))
disp(max(rates))
disp(min(rates))