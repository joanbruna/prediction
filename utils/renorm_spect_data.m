function out=renorm_spect_data(in, stds)
out=in;

out = out./repmat(stds,1,size(out,2));
%norms = sqrt(sum(abs(out).^2)) + 1;
%out = out./repmat(norms,size(out,1),1);


