
options.N=256;
options.L=2^15;
options.Ksmooth=32;
options.ntemplates=2;

[X, temps, phaschange] = generate_jitter_data(options);

X=abs(X);
X=X./repmat(sqrt(sum(X.^2)),size(X,1),1);


ld_param = struct;
%param.D = Dini;
ld_param.K = 500;
ld_param.lambda = 0.1;
ld_param.mu = 10;
ld_param.epochs = 1;
ld_param.batchsize = 100;
ld_param.renorm_input = 0;

[D_ld,A_ld] = nmf_linear_dynamic(X, ld_param);test_denoising_semisup.m
