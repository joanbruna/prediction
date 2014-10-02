function out=scattbatchpredict(SXin, init, options)

niters=100;
lr=4e-2;
state=0;
rho=0.9;

filters = options.filters;
%filters = add_realimag_filters(filters);
J=size(filters.psi{1},2);
border=getoptions(options,'border',2^J*8);

out=init;
S0 = scattbatch(out, options);
nnin = norm(SXin(:))^2;
for n=1:niters

[dx,errore] = compute_batchscatt_grad(out,SXin,filters,border);
if 0
%sgd
out = out - lr * dx ;
else
%momentum
state = rho * state - lr * dx;
out = out + state;
end
fprintf('error is %f \n',errore/nnin)

end

end


function [gout,currerr] = compute_batchscatt_grad(in, Starget, filters,border)


finput=fft(in);
J=size(filters.psi{1},2);
[M,L]=size(Starget);
Nbis = M/(J+1);
currerr=0;

gout=0*in;
%lowpass
rien = repmat(filters.phi{1},1,L);
tmp = ifft(finput.*rien);
gin=tmp(border+1:end-border,:)-Starget(1:Nbis,:);
currerr = currerr + norm(gin(:))^2;
gin=padarray(gin,[border 0]);
gout = ifft(fft(gin).*conj(rien));

%bandpasses
for j=1:J

tmp = ifft(finput.*repmat(filters.psi{1}{j}{1},1,L));
gin = abs(tmp(border+1:end-border,:))-Starget(1+j*Nbis:(j+1)*Nbis,:);
currerr = currerr + norm(gin(:))^2;
gin = padarray(gin,[border 0]);
aux= (gin .* tmp)./(eps+abs(tmp));
gout = gout + real(ifft(fft(aux).*repmat(conj(filters.psi{1}{j}{1}),1,L)));

end


end
