function filts = create_secfilters(Neff, J)


filt_opt.J=J;
filt_opt.filter_format='fourier';
filt_opt.min_margin=0;
filt_opt.boundary='per';
filters = morlet_filter_bank_1d(Neff, filt_opt);
J=length(filters.psi.filter);
for j=1:J
filts{j} = gpuArray(single(filters.psi.filter{j}));
end
filts{J+1} = gpuArray(single(filters.phi.filter));


